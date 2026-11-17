# 两个触控板视频的动作—结果素材

2026-11-17 已从 [Sebtech](https://www.youtube.com/watch?v=lMqXhcpIXSc) 和 [Show Me on Mac](https://www.youtube.com/watch?v=tDa8m3euZU8) 的两个1080p源视频提取14组素材，共178.2秒源区间。13组包含可见运动及对应屏幕画面，1组静态教学卡单独标记 `reference_only`。每组保留 `hand.mp4`、`result.mp4`、`paired.mp4`、来源/时间/crop/hash manifest和回看缩略图；42个媒体文件已核哈希，全部手部与结果片的帧数对应。

第一阶段优先参考两段单指点按。Sebtech片段22.8–26.5秒包含单指动作及App Store窗口打开；Show Me on Mac片段58.5–64.5秒包含教学演示中的点按与照片选择变化。前者是作者手部画中画与屏幕录像，后者是系统教学演示。后者的字幕可能遮挡局部，手也会抬起离开画面；不能假定每帧都有可用的手姿。

其他片段包括滚动、旋转/缩放和切换操作，部分保留混合动作。章节名、可见动作和实际结果分别记录；没有把旋转强行标成纯缩放。裁剪保持原始共同时间轴，不推定原制作者的手部与屏幕采集有传感器级同步。手部片包含触控板与局部背景，尚未生成逐像素手部分割或可靠3D姿态。

这些素材的机器人动作BC资格目前为0：缺少机器人关节目标、实际接触和力真值，尚未完成重定向与物理执行验证。所有manifest的 `robot_action_labels_available` 和 `training_eligible_for_robot_action_bc` 均为false。转换与后续纠错/RL规则见 [视频到BC规范](video-to-bc.zh-CN.md)。

## 复现

原视频与裁剪文件只保存在本地素材库，不随公开仓库分发。将合法取得的两个原视频放入自选素材目录的 `DOWNLOADS/`，文件名对应 [元数据选段表](../tools/sharpa_video_episodes.json)。需要Python、ffmpeg/ffprobe；图库生成还需要Pillow。

```bash
python tools/sharpa_video_cut.py --root /path/to/materials --spec tools/sharpa_video_episodes.json
python tools/sharpa_video_report.py --root /path/to/materials
```

裁剪器支持 `--only <episode_id>` 仅重做选定片段。默认重新编码；`--metadata-only` 必须验证来源文件字节、区间、请求/有效裁剪和已存媒体hash一致才允许更新说明，不能把改变后的选段表贴到旧视频上。有效crop尺寸须与输出实际尺寸一致。先验证媒体，再原子发布manifest。

素材库的 `index.html` 用于离线回看，`materials-index.json` 用于后续提取管线。索引中的“配对”表示来源视频中的可见对应关系，不等于已经训练出机器人策略。
