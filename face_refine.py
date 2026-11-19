# SPDX-License-Identifier: GPL-3.0-or-later
"""Artfat H3 Face Refine: sharpen small / far faces in a finished MiniMax H3 video, chunk by chunk.

Same pipeline as the hand-built v3 graph that passed the user's tests on 18-19 Nov 2026:
  track + crop the face (H3FaceTrackCrop, 768 canvas) -> SAM3 "face" mask on the crops (threshold 0.15,
  all objects, single-frame gaps filled) -> mask cleanup + mask into the H3 latent grid (MaskVidExperiments)
  -> masked latent of the crops + the ORIGINAL audio latent (kept, which is what keeps lip-sync) ->
  DifferentialDiffusion + one H3 Ref2V pass (denoise 0.5, uni_pc / beta / 8 by default) -> decode ->
  VOSR2 polish (optional) -> skin-only tone match -> face-only stitch with the SAM masks.

What is different from the graph is memory: the prompt is encoded once and the text encoder is dropped,
the video is read and written in chunks (lengths on the H3 latent grid, overlaps cross-faded), finished
frames go straight to ffmpeg, so the peak does not grow with the length of the video.
"""
from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.samplers
import comfy.utils
import folder_paths

from .nodecall import call_node, has_node
from .tone import ArtfatFaceToneMatch

NAME = "Artfat H3 Face Refine"


# ----------------------------------------------------------------------------------------- video io
def _probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"{NAME}: cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def _read_frames(path, start, count):
    """Frames [start, start+count) as float32 [count,H,W,3] 0..1 (RGB)."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(count):
        ok, f = cap.read()
        if not ok:
            break
        out.append(torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not out:
        raise RuntimeError(f"{NAME}: no frames read at {start}")
    return torch.stack(out).float().div_(255.0)


def _load_audio(path):
    """Whole audio track as ComfyUI AUDIO, or None when the file has no audio."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        return {"waveform": wav.unsqueeze(0), "sample_rate": sr}
    except Exception:
        pass
    try:  # decode through ffmpeg when torchaudio can't read the container
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", "48000", "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        if not raw:
            return None
        a = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).view(-1, 2).t().contiguous()
        return {"waveform": a.unsqueeze(0), "sample_rate": 48000}
    except Exception:
        return None


def _audio_slice(audio, fps, start, count):
    if audio is None:
        return None
    sr = audio["sample_rate"]
    a = int(round(start / fps * sr))
    b = int(round((start + count) / fps * sr))
    w = audio["waveform"][..., a:b]
    need = b - a
    if w.shape[-1] < need:  # pad silence past the end of the track
        w = F.pad(w, (0, need - w.shape[-1]))
    return {"waveform": w.contiguous(), "sample_rate": sr}


def _video_input_path(video):
    """File path behind a ComfyUI VIDEO input (Load Video). In-memory uploads are written to a temp file."""
    src = video.get_stream_source() if hasattr(video, "get_stream_source") else None
    if isinstance(src, str):
        return src
    if src is not None and hasattr(src, "read"):
        tmp_dir = os.path.join(folder_paths.get_temp_directory(), "artfat_face_refine")
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, f"input_{int(time.time() * 1000)}.mp4")
        src.seek(0)
        with open(path, "wb") as f:
            f.write(src.read())
        return path
    if hasattr(video, "save_to"):
        tmp_dir = os.path.join(folder_paths.get_temp_directory(), "artfat_face_refine")
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, f"input_{int(time.time() * 1000)}.mp4")
        video.save_to(path)
        return path
    raise RuntimeError(f"{NAME}: cannot get a file from the connected video input")


# ----------------------------------------------------------------------------------------- chunk plan
def _grid_len(n):
    """Largest H3-friendly length <= n (frame 0 alone, then groups of 4: 1 + 4k)."""
    return max(1, ((n - 1) // 4) * 4 + 1)


def _plan_chunks(total, chunk, overlap):
    """[(start, length)] covering [0,total); every length is 1+4k; neighbours overlap by `overlap` frames."""
    chunk = _grid_len(max(chunk, 9))
    if total <= chunk:
        return [(0, total)]
    overlap = max(0, min(overlap, chunk // 2))
    # balanced: the fewest chunks of at most `chunk` frames, all about the same length (no tiny tail)
    nc = math.ceil((total - overlap) / (chunk - overlap))
    length = (total + (nc - 1) * overlap + nc - 1) // nc
    length = min(chunk, ((length - 1 + 3) // 4) * 4 + 1)
    spans, s = [], 0
    while True:
        if s + length >= total:
            spans.append((s, total - s))
            break
        spans.append((s, length))
        s += length - overlap
    return spans


def _pad_to_grid(frames):
    """Repeat the last frame so the count is 1+4k; returns (frames, real_count)."""
    n = frames.shape[0]
    m = ((n - 1 + 3) // 4) * 4 + 1
    if m == n:
        return frames, n
    return torch.cat([frames, frames[-1:].expand(m - n, -1, -1, -1)], 0), n


def _fill_mask_gaps(masks, min_pixels=16):
    """Frames whose face mask is empty borrow the nearest non-empty frame's mask (SAM3 drops single frames)."""
    filled = masks.clone()
    has = (masks > 0.5).flatten(1).sum(1) >= min_pixels
    idx = has.nonzero().flatten().tolist()
    if not idx:
        return filled, masks.shape[0]
    gaps = 0
    for i in range(masks.shape[0]):
        if not bool(has[i]):
            j = min(idx, key=lambda k: abs(k - i))
            filled[i] = masks[j]
            gaps += 1
    return filled, gaps


# ----------------------------------------------------------------------------------------- the node
def _clip_names():
    names = list(folder_paths.get_filename_list("text_encoders"))
    try:
        names += [n for n in folder_paths.get_filename_list("clip_gguf") if n not in names]
    except Exception:
        pass
    return sorted(set(names))


class ArtfatH3FaceRefine:
    @classmethod
    def INPUT_TYPES(cls):
        samplers = comfy.samplers.KSampler.SAMPLERS
        schedulers = comfy.samplers.KSampler.SCHEDULERS
        ckpts = folder_paths.get_filename_list("checkpoints")
        sam_default = next((c for c in ckpts if "sam3" in c.lower()), ckpts[0] if ckpts else "")
        clips = _clip_names()
        clip_default = next((c for c in clips if "qwen3vl_32b_minimax_h3" in c), clips[0] if clips else "")
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "tooltip": "Full path to the finished H3 video (mp4). Used only when the 'video' input is not connected. Its audio is kept and also guides the mouth."}),
                "model": ("MODEL", {"tooltip": "The H3 model you generated with (e.g. Pika), shift already applied (MiniMaxH3SigmaShift 12/3)."}),
                "vae": ("VAE", {"tooltip": "H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "H3 audio VAE."}),
                "text_encoder": (clips, {"default": clip_default, "tooltip": "Loaded inside the node, used once for the prompt, then dropped (saves ~15 GB)."}),
                "sam3_checkpoint": (ckpts, {"default": sam_default, "tooltip": "SAM 3.1 checkpoint for the face mask."}),
                "prompt": ("STRING", {"multiline": True, "default": "subject_definitions:\n<Subject 1> is the woman with the face from <Picture 1>.\nsummary:\n[Reference-to-video] Close view of her face.\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - her face, eyes, lips, hair of <Picture 1>.\ndetailed_description:\n[Shot 1] Her face exactly as in <Picture 1>, sharp natural eyes, lashes and lips, real skin with pores. Her expression, speech and head movement exactly as in the input clip. [Static shot]\noverall_soundscape:\nAs in the input clip.\nnon_diegetic_music:\nN/A"}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05, "tooltip": "0.5 tested best. Lower keeps the expression closer, higher redraws more."}),
                "sampler_name": (samplers, {"default": "uni_pc" if "uni_pc" in samplers else samplers[0]}),
                "scheduler": (schedulers, {"default": "beta" if "beta" in schedulers else schedulers[0]}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sam_threshold": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.95, "step": 0.01, "tooltip": "0.15 tested: 0.3 lost the face on some frames and the face jiggled."}),
                "vosr_polish": ("BOOLEAN", {"default": True, "tooltip": "VOSR2 polish of the refined crop (needs ComfyUI_LC_MaskMaker_nodes)."}),
                "tone_match": ("BOOLEAN", {"default": True, "tooltip": "Skin-only tone correction toward the original (the refine pass darkens and cools skin)."}),
                "chunk_frames": ("INT", {"default": 145, "min": 33, "max": 481, "step": 4, "tooltip": "Frames per chunk (rounded to 1+4k). Memory does not grow with video length."}),
                "overlap_frames": ("INT", {"default": 24, "min": 0, "max": 96, "step": 4, "tooltip": "Overlap between chunks, cross-faded."}),
                "filename_prefix": ("STRING", {"default": "FaceRefine/refined"}),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "Connect ComfyUI's Load Video node (pick or upload a file). The node only takes the file from it and still reads the video in chunks."}),
                "face_ref": ("IMAGE", {"tooltip": "Face picture of the person. Empty = the face crop of the first frame (tested slightly better than a neutral portrait)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "report")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Artfat/H3"

    async def run(self, video_path, model, vae, audio_vae, text_encoder, sam3_checkpoint, prompt, denoise,
                  sampler_name, scheduler, steps, seed, sam_threshold, vosr_polish, tone_match, chunk_frames,
                  overlap_frames, filename_prefix, video=None, face_ref=None):
        t0 = time.time()
        if video is not None:
            video_path = _video_input_path(video)
        video_path = video_path.strip().strip('"')
        if not os.path.isfile(video_path):
            raise RuntimeError(f"{NAME}: video not found: {video_path}")
        if vosr_polish and not has_node("LCVOSRUpscale"):
            raise RuntimeError(f"{NAME}: vosr_polish needs ComfyUI_LC_MaskMaker_nodes (LCVOSRUpscale); switch it off or install it.")
        fps, total, W, H = _probe(video_path)
        audio = _load_audio(video_path)
        spans = _plan_chunks(total, chunk_frames, overlap_frames)
        log = [f"{NAME}: {total} frames {W}x{H} @ {fps:.2f} fps, {len(spans)} chunk(s) {spans}"]
        print("[ArtfatFaceRefine] " + log[0])

        # output: frames straight into ffmpeg, original audio muxed in
        out_dir = folder_paths.get_output_directory()
        full_prefix, subfolder = os.path.split(filename_prefix)
        full_dir = os.path.join(out_dir, full_prefix)
        os.makedirs(full_dir, exist_ok=True)
        stem = subfolder or "refined"
        k = 1
        while os.path.exists(os.path.join(full_dir, f"{stem}_{k:05d}.mp4")):
            k += 1
        out_path = os.path.join(full_dir, f"{stem}_{k:05d}.mp4")
        ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", f"{fps}", "-i", "-", "-i", video_path, "-map", "0:v", "-map", "1:a?",
                               "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", out_path],
                              stdin=subprocess.PIPE)

        def emit(frames_u8):
            ff.stdin.write(frames_u8.contiguous().numpy().tobytes())

        # SAM3 model + "face" conditioning, loaded once
        sam_model, sam_clip = (await call_node("CheckpointLoaderSimple", ckpt_name=sam3_checkpoint))[:2]
        sam_cond = (await call_node("CLIPTextEncode", clip=sam_clip, text="face"))[0]

        cond = None
        ref_img = face_ref
        pbar = comfy.utils.ProgressBar(len(spans))
        prev_tail = None  # last `overlap` refined frames of the previous chunk (uint8), for the cross-fade
        written = 0
        gaps_total = 0
        try:
            for ci, (start, count) in enumerate(spans):
                mm.throw_exception_if_processing_interrupted()
                frames = _read_frames(video_path, start, count)
                frames, real = _pad_to_grid(frames)
                n = frames.shape[0]

                # 1) track + crop (no identity model: single person, continuity tracking)
                tc = await call_node("H3FaceTrackCrop", images=frames, detector="face_yolov8m.pt", confidence=0.35,
                                     crop_factor=2.5, canvas_width=768, canvas_height=768, canvas_mode="manual",
                                     smooth_window=21, size_smooth_window=51, smooth_method="gaussian",
                                     size_mode="per_frame", identity_track=False, select="largest_face")
                crops, transform, cw, ch = tc[0], tc[1], int(tc[4]), int(tc[5])
                if ref_img is None:
                    ref_img = crops[:1].clone()

                # 2) SAM3 face mask on the crops, gaps filled
                track = (await call_node("SAM3_VideoTrack", images=crops, model=sam_model, conditioning=sam_cond,
                                         detection_threshold=sam_threshold, max_objects=0, detect_interval=1))[0]
                raw_masks = (await call_node("SAM3_TrackToMask", track_data=track, object_indices=""))[0]
                del track
                raw_masks, gaps = _fill_mask_gaps(raw_masks)
                gaps_total += gaps
                clean = (await call_node("MVEx_MaskCleanup", masks=raw_masks, threshold=0.3, method="shrink_grow",
                                         **{"method.shrink": 12, "method.min_frames": 4}, edge_grow=16))[0]
                lat_mask = (await call_node("MVEx_MaskToLatentSpace", masks=clean, compression="auto",
                                            spatial_method="max", temporal_method="max", grow_spatial=36,
                                            grow_temporal=1, grow_temp_mode="both", vae=vae))[0]

                # 3) conditioning: encode the prompt ONCE, then drop the text encoder for the rest of the run
                if cond is None:
                    ref32 = (await call_node("ResizeImageMaskNode", input=ref_img, resize_type="scale to multiple",
                                             **{"resize_type.multiple": 32}, scale_method="area"))[0]
                    if text_encoder.lower().endswith(".gguf"):
                        clip = (await call_node("CLIPLoaderGGUF", clip_name=text_encoder, type="minimax"))[0]
                    else:
                        clip = (await call_node("CLIPLoader", clip_name=text_encoder, type="minimax", device="default"))[0]
                    cond = (await call_node("MiniMaxH3ReferenceToVideo", clip=clip, prompt=prompt, width=cw, height=ch,
                                            length=n, ref_image_size="match", vae=vae, audio_vae=audio_vae,
                                            **{"ref_images.ref_image_0": ref32}))[0]
                    del clip
                    gc.collect()
                    mm.unload_all_models()
                    mm.soft_empty_cache()

                # 4) masked video latent of the crops + kept original audio latent
                vlat = (await call_node("VAEEncode", pixels=crops, vae=vae))[0]
                vlat = (await call_node("SetLatentNoiseMask", samples=vlat, mask=lat_mask))[0]
                a = _audio_slice(audio, fps, start, n) if audio is not None else {"waveform": torch.zeros(1, 2, int(n / fps * 48000)), "sample_rate": 48000}
                alat = (await call_node("VAEEncodeAudio", audio=a, vae=audio_vae))[0]
                zero = (await call_node("SolidMask", value=0.0, width=32, height=32))[0]
                alat = (await call_node("SetLatentNoiseMask", samples=alat, mask=zero))[0]
                av = (await call_node("LTXVConcatAVLatent", video_latent=vlat, audio_latent=alat))[0]
                del vlat, alat

                # 5) the face pass
                dd = (await call_node("DifferentialDiffusion", model=model, strength=1.0))[0]
                guider = (await call_node("BasicGuider", model=dd, conditioning=cond))[0]
                sampler = (await call_node("KSamplerSelect", sampler_name=sampler_name))[0]
                sigmas = (await call_node("BasicScheduler", model=model, scheduler=scheduler, steps=steps, denoise=denoise))[0]
                noise = (await call_node("RandomNoise", noise_seed=seed))[0]
                out = (await call_node("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler,
                                       sigmas=sigmas, latent_image=av))[0]
                del av, dd, guider
                v_out = (await call_node("LTXVSeparateAVLatent", av_latent=out))[0]
                del out
                refined = (await call_node("VAEDecode", samples=v_out, vae=vae))[0]
                del v_out

                # 6) polish + tone
                if vosr_polish:
                    up = (await call_node("LCVOSRUpscale", image=refined, upscale_by="1.5x", seed=42,
                                          color_alignment="wavelet", tile_size=512, vae_tile_size=1024))[0]
                    refined = (await call_node("ImageScale", image=up, upscale_method="lanczos", width=cw, height=ch, crop="disabled"))[0]
                    del up
                if tone_match:
                    refined = ArtfatFaceToneMatch().match(crops, refined, raw_masks, 1.0, 9, 40.0, 90.0, 4)[0]

                # 7) stitch the face back
                stitched = (await call_node("H3FaceStitch", base_images=frames, refined_crops=refined, transform=transform,
                                            paste_region="face_only", mask_dilation=24, feather=24, colour_match=1.0,
                                            blend=1.0, undetected_frames="fade_out", masks=raw_masks))[0]
                del refined, crops, frames, raw_masks, clean, lat_mask
                stitched = stitched[:real]
                u8 = (stitched.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu()
                del stitched

                # 8) cross-fade the overlap with the previous chunk, write, keep this chunk's tail
                ov = 0 if ci == 0 else spans[ci - 1][0] + spans[ci - 1][1] - start
                if prev_tail is not None and ov > 0:
                    ov = min(ov, prev_tail.shape[0], u8.shape[0])
                    w = torch.linspace(0, 1, ov + 2)[1:-1].view(-1, 1, 1, 1)
                    mix = (prev_tail[-ov:].float() * (1 - w) + u8[:ov].float() * w + 0.5).to(torch.uint8)
                    emit(mix)
                    written += ov
                    body = u8[ov:]
                else:
                    body = u8
                nxt_ov = 0
                if ci + 1 < len(spans):
                    nxt_ov = start + real - spans[ci + 1][0]
                keep = body[: body.shape[0] - nxt_ov] if nxt_ov > 0 else body
                emit(keep)
                written += keep.shape[0]
                prev_tail = u8[-nxt_ov:].clone() if nxt_ov > 0 else None
                del u8, body, keep
                gc.collect()
                mm.soft_empty_cache()
                pbar.update(1)
                log.append(f"chunk {ci + 1}/{len(spans)} frames {start}-{start + real - 1} done, mask gaps filled {gaps}")
                print("[ArtfatFaceRefine] " + log[-1])
        finally:
            try:
                ff.stdin.close()
            except Exception:
                pass
            ff.wait()
        log.append(f"written {written}/{total} frames, mask gaps filled {gaps_total}, {time.time() - t0:.0f} s -> {out_path}")
        print("[ArtfatFaceRefine] " + log[-1])
        rel = os.path.relpath(out_path, out_dir)
        sub, fname = os.path.split(rel)
        return {"ui": {"gifs": [{"filename": fname, "subfolder": sub, "type": "output", "format": "video/h264-mp4"}]},
                "result": (out_path, "\n".join(log))}


NODE_CLASS_MAPPINGS = {"ArtfatH3FaceRefine": ArtfatH3FaceRefine}
NODE_DISPLAY_NAME_MAPPINGS = {"ArtfatH3FaceRefine": "Artfat H3 Face Refine (chunked)"}
