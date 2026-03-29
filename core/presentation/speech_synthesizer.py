import asyncio
import edge_tts
from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip
from typing import List, Optional
import os

class SpeechSynthesizer:
    def __init__(self, voice: str = "en-US-JennyNeural"):
        self.voice = voice

    async def synthesize(self, text: str, output_audio: str):
        """生成语音文件"""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_audio)

    def create_video(self, audio_path: str, slide_images: List[str],
                     timings: Optional[List[float]] = None,
                     output_video: str = "output.mp4", fps=24, size=(1920,1080)):
        """
        将音频和幻灯片图片合成为视频。
        :param audio_path: 音频文件路径
        :param slide_images: 幻灯片图片路径列表
        :param timings: 每张幻灯片的开始时间（秒），若不提供则均匀分配
        :param output_video: 输出视频路径
        :param fps: 视频帧率
        :param size: 视频尺寸
        """
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        num_slides = len(slide_images)

        if timings is None:
            # 均匀分配
            slide_duration = total_duration / num_slides
            timings = [i * slide_duration for i in range(num_slides)]

        clips = []
        for i, img_path in enumerate(slide_images):
            start = timings[i]
            end = timings[i+1] if i+1 < len(timings) else total_duration
            clip = ImageClip(img_path).set_duration(end - start).set_start(start)
            clips.append(clip)

        video = CompositeVideoClip(clips, size=size).set_audio(audio)
        video.write_videofile(output_video, fps=fps, codec='libx264', audio_codec='aac')