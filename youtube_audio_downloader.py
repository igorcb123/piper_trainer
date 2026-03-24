import argparse
from pathlib import Path

import yt_dlp


def download_audio(url: str, output_dir: Path, audio_format: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base_name = ydl.prepare_filename(info)
        downloaded_path = Path(base_name).with_suffix(f".{audio_format}")

    return downloaded_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download audio from a YouTube URL for later transcription with Whisper."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("downloads"),
        help="Folder where the audio will be saved",
    )
    parser.add_argument(
        "-f",
        "--format",
        default="mp3",
        choices=("mp3", "m4a", "wav", "opus", "flac"),
        help="Audio format to extract",
    )
    args = parser.parse_args()

    audio_path = download_audio(args.url, args.output_dir, args.format)
    print(f"Audio guardado en: {audio_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
