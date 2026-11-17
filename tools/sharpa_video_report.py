#!/usr/bin/env python3
"""Build offline gallery + materials index + Chinese report for SHARPA episodes.

Usage: python sharpa_video_report.py --root OUTPUT_DIR
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{p.stderr[-1500:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="output root containing episodes/")
    args = ap.parse_args()
    root = Path(args.root)
    ep_root = root / "episodes"
    eps = []
    for d in sorted(ep_root.iterdir()):
        m = d / "manifest.json"
        if not m.exists():
            continue
        man = json.loads(m.read_text())
        eps.append((d, man))

    # contact sheet + thumbnail per episode (from hand.mp4)
    for d, man in eps:
        sheet = d / "contact.jpg"
        if not sheet.exists():
            run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "hand.mp4"),
                 "-vf", "fps=1/3,scale=320:-1,tile=3x3:padding=3:margin=3",
                 "-frames:v", "1", str(sheet)])
        thumb = d / "thumb.jpg"
        if not thumb.exists():
            dur = man["hand"]["duration_s"]
            run(["ffmpeg", "-v", "error", "-y", "-ss", f"{dur/2:.1f}",
                 "-i", str(d / "hand.mp4"), "-frames:v", "1", "-vf",
                 "scale=320:-1", str(thumb)])

    index = {
        "generated_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                        capture_output=True, text=True).stdout.strip(),
        "root": str(ep_root),
        "robot_action_labels_available": False,
        "training_eligible_for_robot_action_bc": False,
        "episodes": [
            {
                "id": man["episode_id"],
                "dir": f"episodes/{d.name}",
                "files": {"hand": "hand.mp4", "result": "result.mp4",
                          "paired": "paired.mp4", "manifest": "manifest.json",
                          "contact": "contact.jpg", "thumb": "thumb.jpg"},
                "labels": man["labels"],
                "source": {"url": man["source"]["url"],
                           "sha256": man["source"]["sha256"]},
                "source_start_s": man["source_start_s"],
                "source_end_s": man["source_end_s"],
                "duration_s": man["paired"]["duration_s"],
                "source_kind": man["source_kind"],
                "provenance": man["provenance"],
                "visible_hand_motion": man["visible_hand_motion"],
                "reference_only": man["reference_only"],
                "observed_result": man["observed_result"],
                "pairing_confidence": man["pairing_confidence"],
                "limitations": man["limitations"],
                "flags": man["flags"],
            } for d, man in eps
        ],
    }
    (root / "materials-index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False))

    cards = []
    for d, man in eps:
        lab = man["labels"]
        cards.append(f"""
<div class="card">
  <h3>{man['episode_id']}</h3>
  <img src="episodes/{d.name}/thumb.jpg" alt="{man['episode_id']}">
  <p><b>动作:</b> {lab.get('gesture','')}<br>
     <b>手势视图:</b> {lab.get('hand_view','')}<br>
     <b>来源:</b> <a href="{man['source']['url']}">{man['source']['url']}</a><br>
     <b>源区间:</b> {man['source_start_s']}s – {man['source_end_s']}s<br>
     <b>时长:</b> {man['paired']['duration_s']}s<br>
     <b>source_kind:</b> {man['source_kind']}<br>
     <b>可见手部运动:</b> {man['visible_hand_motion']} |
     <b>配对置信度:</b> {man['pairing_confidence']} |
     <b>仅参考:</b> {man['reference_only']}<br>
     <b>观察到的结果:</b> {man['observed_result']}<br>
     <b>局限:</b> {man['limitations']}<br>
     <b>机器人动作标签:</b> 不可用（robot_action_labels_available=false）<br>
     <b>裁剪框(hand 实际):</b> {man['hand']['crop_xywh_effective']} <b>(result):</b> {man['result']['crop_xywh_effective']}</p>
  <video controls preload="metadata" src="episodes/{d.name}/paired.mp4"></video>
  <p class="links">
    <a href="episodes/{d.name}/hand.mp4">hand.mp4</a> |
    <a href="episodes/{d.name}/result.mp4">result.mp4</a> |
    <a href="episodes/{d.name}/paired.mp4">paired.mp4</a> |
    <a href="episodes/{d.name}/manifest.json">manifest.json</a> |
    <a href="episodes/{d.name}/contact.jpg">contact.jpg</a>
  </p>
</div>""")
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>SHARPA Trackpad 视频材料索引</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:2em;background:#fafafa}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1em;margin:1em 0;max-width:980px}}
video{{max-width:100%;margin-top:.5em}}
img{{max-width:320px}}
.links a{{margin-right:.5em}}
.meta{{color:#666;font-size:.9em}}
</style></head><body>
<h1>SHARPA Trackpad 公开视频手势材料（2026-11-17）</h1>
<p class="meta">共 {len(eps)} 个 episode，其中 {sum(1 for _,m in eps if not m['reference_only'])} 个为运动配对剪辑，{sum(1 for _,m in eps if m['reference_only'])} 个为仅参考静态卡片。全部来自两个公开 YouTube 视频；仅为参考材料。
机器人动作标签不可用（robot_action_labels_available=false），不可用于机器人动作 BC 训练（training_eligible_for_robot_action_bc=false）。
源 A 为创作者真人手部 PiP 实拍+屏幕录制；源 B 为 macOS 内置教学演示（含运动演示片段与一张静态参考卡片），不得据此推断实时控制或真实相机因果关系。</p>
{''.join(cards)}
</body></html>"""
    (root / "index.html").write_text(html)

    # REPORT.md (Chinese)
    lines = ["# SHARPA Trackpad 视频提取报告（2026-11-17）", "",
             "## 来源", "",
             "| 源 | URL | 上传者 | 日期 | 时长 | 本地文件 | SHA256(前16) |", "|---|---|---|---|---|---|---|"]
    seen = {}
    for d, man in eps:
        s = man["source"]
        if s["url"] not in seen:
            seen[s["url"]] = man
    src_dur = {}
    for url, man in seen.items():
        f = root / "DOWNLOADS" / man["source"]["file"]
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout.strip()
        src_dur[url] = round(float(out), 1) if out else "?"
    for url, man in seen.items():
        s = man["source"]
        lines.append(f"| {s['uploader']} | {url} | {s['uploader']} | {s.get('upload_date','?')} | "
                     f"{src_dur[url]}s | {s['file']} | {s['sha256'][:16]}… |")
    moving = [m for _, m in eps if not m["reference_only"]]
    refonly = [m for _, m in eps if m["reference_only"]]
    lines += ["", f"## Episode 清单（运动配对 {len(moving)} 个；仅参考静态卡片 {len(refonly)} 个）", "",
              "| ID | 手势 | source_kind | 源区间(s) | 时长(s) | hand实际裁剪 | result实际裁剪 |", "|---|---|---|---|---|---|---|"]
    for d, man in eps:
        lab = man["labels"]
        lines.append(f"| {man['episode_id']} | {lab.get('gesture','')} | {man['source_kind']} | "
                     f"{man['source_start_s']}–{man['source_end_s']} | {man['paired']['duration_s']} | "
                     f"{man['hand']['crop_xywh_effective']} | {man['result']['crop_xywh_effective']} |")
    lines += ["", "## 重要说明", "",
              "- 源 A（Sebtech，德语）：真人手部+触控板画中画（PiP）实拍，与屏幕共享同一时间轴，视图同步。",
              "- 源 B（Show Me on Mac，英语）：macOS 内置触控板教学演示（source_kind=os_instructional_demonstration）。其中 Tap to Click / Secondary click / Zoom / Scroll / Swipe Between Pages 章节含运动演示片段（producer 合成，手-屏对齐未传感器核验）；Launchpad 为静态图示卡片（reference_only，不计入运动配对）。",
              "- 手部缺失/不稳定区间：源 A 的转场、PiP 淡出及固定裁剪框外区间未纳入或已在各 manifest limitations 中注明（如 a01 的 2 指段、a02 结尾、a03 混合旋转+缩放）。",
              "- 所有剪辑保留原始时间（无人工时间偏移）；hand/result/paired 使用相同源帧。",
              "- 所有产出片段均经 ffmpeg 解码校验与 ffprobe 帧数/时长检查（见各 manifest.json）。",
              "- 普通视频仅提供参考材料：无 3D、无机器人关节、无力/接触真值。robot_action_labels_available=false，training_eligible_for_robot_action_bc=false（源参考待重定向）。",
              "- 下载：本机 yt-dlp 被 YouTube GVS 返回 HTTP 403 阻断，后经文内档记录的授权备用主机普通下载获取两个 1080p 源（合计约 90MB，远低于 2GB 上限）；具体主机为私有信息，见私有 PROGRESS.json。",
              "- 方法与复现：见 SOURCES_METHODS.md。",
              ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(f"episodes={len(eps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
