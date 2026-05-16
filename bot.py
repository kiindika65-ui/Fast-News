import asyncio
import hashlib
import html
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import edge_tts
import feedparser
import numpy as np
import requests
from moviepy.editor import AudioFileClip, VideoClip
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# ===================== SETTINGS =====================

PAGE_NAME = os.getenv("PAGE_NAME", "World Pulse Daily")
VOICE = os.getenv("VOICE", "en-US-GuyNeural")

VIDEO_SECONDS_MIN = int(os.getenv("VIDEO_SECONDS_MIN", "35"))
VIDEO_SECONDS_MAX = int(os.getenv("VIDEO_SECONDS_MAX", "55"))
MAX_OLD_ITEMS = int(os.getenv("MAX_OLD_ITEMS", "500"))
MAX_VIDEOS_PER_RUN = int(os.getenv("MAX_VIDEOS_PER_RUN", "1"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
STATE_FILE = Path(os.getenv("STATE_FILE", "state.json"))

FPS = int(os.getenv("FPS", "24"))
W, H = 1080, 1920

RSS_FEEDS = [
    "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/headlines/section/topic/NATION?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/headlines/section/topic/HEALTH?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-US&gl=US&ceid=US:en",
    "https://feeds.npr.org/1001/rss.xml",
    "https://feeds.npr.org/1003/rss.xml",
    "https://abcnews.go.com/abcnews/topstories",
    "https://www.cbsnews.com/latest/rss/main",
    "https://feeds.nbcnews.com/nbcnews/public/news",
]

BANNED_TITLE_WORDS = [
    "live updates",
    "opinion",
    "analysis:",
    "sponsored",
    "advertisement",
    "newsletter",
]


# ===================== FIX OUTPUT FOLDER ERROR =====================

def prepare_folders():
    """
    Fixes this error:
    FileExistsError: [Errno 17] File exists: 'output'

    That happens when 'output' exists as a file, not a folder.
    """
    if OUTPUT_DIR.exists() and not OUTPUT_DIR.is_dir():
        OUTPUT_DIR.unlink()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===================== STATE =====================

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"used": []}

    return {"used": []}


def save_state(state: dict) -> None:
    state["used"] = state.get("used", [])[-MAX_OLD_ITEMS:]

    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ===================== TEXT HELPERS =====================

def clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def source_from_entry(entry) -> str:
    source = ""

    if hasattr(entry, "source") and isinstance(entry.source, dict):
        source = entry.source.get("title", "")

    if not source:
        source = getattr(entry, "author", "") or getattr(entry, "publisher", "")

    if not source:
        link = getattr(entry, "link", "")
        host = urlparse(link).netloc.replace("www.", "")
        source = host.split(".")[0].title() if host else "RSS News"

    return clean_text(source)[:40]


def item_id(title: str, link: str) -> str:
    base = re.sub(r"[^a-z0-9]+", " ", (title + " " + link).lower()).strip()
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def is_good_item(title: str, summary: str) -> bool:
    title_lower = title.lower()

    if len(title) < 25 or len(title) > 190:
        return False

    if any(word in title_lower for word in BANNED_TITLE_WORDS):
        return False

    return True


# ===================== RSS =====================

def fetch_items() -> list[dict]:
    items = []
    feeds = RSS_FEEDS[:]
    random.shuffle(feeds)

    headers = {
        "User-Agent": "WorldPulseDailyBot/1.0"
    }

    for feed_url in feeds:
        try:
            response = requests.get(feed_url, headers=headers, timeout=20)
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
        except Exception as e:
            print(f"Feed failed: {feed_url} -> {e}")
            continue

        for entry in parsed.entries[:20]:
            title = clean_text(getattr(entry, "title", ""))
            summary = clean_text(getattr(entry, "summary", ""))
            link = clean_text(getattr(entry, "link", ""))
            source = source_from_entry(entry)

            if not is_good_item(title, summary):
                continue

            items.append({
                "id": item_id(title, link),
                "title": title,
                "summary": summary,
                "link": link,
                "source": source,
                "feed": feed_url,
            })

    random.shuffle(items)
    return items


def pick_new_items(items: list[dict], state: dict, count: int) -> list[dict]:
    used = set(state.get("used", []))
    picked = []
    seen_titles = set()

    for item in items:
        title_key = re.sub(
            r"[^a-z0-9]+",
            " ",
            item["title"].lower()
        ).strip()[:90]

        if item["id"] in used:
            continue

        if title_key in seen_titles:
            continue

        seen_titles.add(title_key)
        picked.append(item)

        if len(picked) >= count:
            break

    return picked


# ===================== ORIGINAL SCRIPT =====================

def make_original_script(item: dict) -> str:
    title = item["title"].strip(" .")
    summary = item.get("summary", "").strip(" .")
    source = item.get("source", "RSS news source")

    if " - " in title:
        possible_headline, possible_source = title.rsplit(" - ", 1)

        if 12 <= len(possible_headline) <= 150:
            title = possible_headline.strip()
            source = possible_source.strip() or source

    short_summary = summary

    if len(short_summary) > 180:
        short_summary = short_summary[:180].rsplit(" ", 1)[0] + "."

    openers = [
        "Here is a fast update from the United States news cycle.",
        "A new headline is developing in U.S. news today.",
        "Here is one of the latest stories people are following right now.",
        "This is a quick news brief from World Pulse Daily.",
        "A new update is now appearing across major news feeds.",
    ]

    middles = [
        f"Reports from {source} point to this main development: {title}.",
        f"The central update is this: {title}.",
        f"According to the latest RSS update from {source}, the story is about {title}.",
        f"The headline now getting attention is: {title}.",
    ]

    context = ""

    if short_summary and short_summary.lower() not in title.lower():
        context = f" In simple terms, {short_summary}"

    closers = [
        "More details may change as officials and reporters update the story.",
        "We will keep watching for verified updates as this develops.",
        "Follow reliable sources before making conclusions, because breaking stories can change quickly.",
        "This is a developing update, and more verified information may come later.",
    ]

    script = f"{random.choice(openers)} {random.choice(middles)}{context} {random.choice(closers)}"
    script = re.sub(r"\s+", " ", script).strip()

    words = script.split()

    if len(words) > 115:
        script = " ".join(words[:115]).rsplit(".", 1)[0] + "."

    return script


# ===================== VOICE =====================

async def make_voice(text: str, out_mp3: Path) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate="+3%",
        volume="+0%"
    )

    await communicate.save(str(out_mp3))


# ===================== VIDEO HELPERS =====================

def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",

        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


def wrap_lines(text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines = []
    line = ""

    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    for word in words:
        test = (line + " " + word).strip()
        box = draw.textbbox((0, 0), test, font=font)

        if box[2] - box[0] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)

            line = word

    if line:
        lines.append(line)

    return lines


def split_caption_chunks(script: str) -> list[str]:
    words = script.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)

        if len(current) >= 8 or word.endswith((".", "?", "!")):
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def draw_round_rect(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(
        xy,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width
    )


def make_background(seed: int) -> Image.Image:
    random.seed(seed)

    base1 = np.array([12, 24, 48], dtype=np.uint8)
    base2 = np.array([35, 54, 88], dtype=np.uint8)

    y = np.linspace(0, 1, H)[:, None]
    grad = (base1 * (1 - y) + base2 * y).astype(np.uint8)

    img = np.repeat(grad[:, None, :], W, axis=1)
    im = Image.fromarray(img, "RGB")

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    for _ in range(16):
        x = random.randint(-200, W)
        y0 = random.randint(-200, H)
        r = random.randint(120, 360)
        alpha = random.randint(18, 48)

        d.ellipse(
            (x, y0, x + r, y0 + r),
            fill=(255, 255, 255, alpha)
        )

    overlay = overlay.filter(ImageFilter.GaussianBlur(22))

    return Image.alpha_composite(
        im.convert("RGBA"),
        overlay
    ).convert("RGB")


def make_frame_builder(item: dict, script: str, duration: float):
    bg = make_background(int(item["id"][:6], 16))

    title_font = get_font(58, True)
    cap_font = get_font(48, True)
    small_font = get_font(34, False)
    tiny_font = get_font(28, False)
    source_font = get_font(36, True)

    title = item["title"]

    if " - " in title:
        title = title.rsplit(" - ", 1)[0]

    title_lines = wrap_lines(title, title_font, 900)[:5]

    captions = split_caption_chunks(script)
    total_chars = sum(max(1, len(c)) for c in captions)

    starts = []
    cursor = 0.0

    for cap in captions:
        starts.append(cursor)
        cursor += duration * (len(cap) / total_chars)

    def caption_at(t: float) -> str:
        idx = 0

        for i, st in enumerate(starts):
            if t >= st:
                idx = i
            else:
                break

        if not captions:
            return ""

        return captions[min(idx, len(captions) - 1)]

    def frame(t: float):
        im = bg.copy().convert("RGBA")
        draw = ImageDraw.Draw(im)

        progress = max(0, min(1, t / max(duration, 0.1)))

        # Top label
        draw_round_rect(
            draw,
            (70, 70, 1010, 155),
            34,
            fill=(255, 255, 255, 32),
            outline=(255, 255, 255, 80),
            width=2
        )

        draw.text(
            (100, 92),
            PAGE_NAME.upper(),
            font=source_font,
            fill=(255, 255, 255, 255)
        )

        draw.text(
            (760, 100),
            "LATEST NEWS",
            font=tiny_font,
            fill=(255, 210, 90, 255)
        )

        # Headline box
        draw_round_rect(
            draw,
            (60, 255, 1020, 760),
            44,
            fill=(0, 0, 0, 105),
            outline=(255, 255, 255, 85),
            width=2
        )

        y = 305

        for line in title_lines:
            draw.text(
                (105, y),
                line,
                font=title_font,
                fill=(255, 255, 255, 255)
            )
            y += 72

        source_text = f"Source: {item.get('source', 'RSS update')}"

        draw.text(
            (105, 705),
            source_text[:48],
            font=small_font,
            fill=(220, 230, 245, 255)
        )

        # Sound bars
        cx = W // 2

        for i in range(8):
            bar_h = int(55 + 45 * np.sin((t * 2.5) + i))
            x = cx - 220 + i * 62

            draw_round_rect(
                draw,
                (x, 880 - bar_h, x + 34, 880 + bar_h),
                14,
                fill=(255, 255, 255, 80)
            )

        # Captions
        cap = caption_at(t)
        cap_lines = wrap_lines(cap, cap_font, 900)[:4]

        panel_top = 1215

        draw_round_rect(
            draw,
            (60, panel_top, 1020, 1620),
            44,
            fill=(0, 0, 0, 145),
            outline=(255, 255, 255, 80),
            width=2
        )

        y = panel_top + 55

        for line in cap_lines:
            draw.text(
                (105, y),
                line,
                font=cap_font,
                fill=(255, 255, 255, 255)
            )
            y += 64

        # Progress bar
        draw_round_rect(
            draw,
            (80, 1740, 1000, 1775),
            18,
            fill=(255, 255, 255, 50)
        )

        draw_round_rect(
            draw,
            (80, 1740, int(80 + 920 * progress), 1775),
            18,
            fill=(255, 210, 90, 230)
        )

        draw.text(
            (95, 1810),
            "Original short news brief • No copied logos • Verify developing stories",
            font=tiny_font,
            fill=(225, 235, 250, 230)
        )

        return np.array(im.convert("RGB"))

    return frame


def safe_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:70] or "news_video"


def create_video(item: dict, script: str, audio_path: Path) -> Path:
    audio = AudioFileClip(str(audio_path))

    duration = min(
        max(audio.duration + 0.6, VIDEO_SECONDS_MIN),
        VIDEO_SECONDS_MAX
    )

    frame_builder = make_frame_builder(item, script, duration)

    clip = VideoClip(
        frame_builder,
        duration=duration
    ).set_audio(
        audio.subclip(0, min(audio.duration, duration))
    )

    filename = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
        f"{safe_filename(item['title'])}.mp4"
    )

    out_path = OUTPUT_DIR / filename

    clip.write_videofile(
        str(out_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=2,
        verbose=False,
        logger=None
    )

    clip.close()
    audio.close()

    return out_path


# ===================== MAIN =====================

async def main():
    prepare_folders()

    state = load_state()

    items = fetch_items()
    picked = pick_new_items(items, state, MAX_VIDEOS_PER_RUN)

    if not picked:
        print("No fresh unused RSS item found.")
        print("Try adding more RSS feeds or clear old state.json.")
        return

    made = []

    for item in picked:
        script = make_original_script(item)

        print("TITLE:", item["title"])
        print("SOURCE:", item["source"])
        print("SCRIPT:", script)

        audio_path = OUTPUT_DIR / f"voice_{item['id']}.mp3"

        await make_voice(script, audio_path)

        video_path = create_video(item, script, audio_path)

        made.append(str(video_path))

        state.setdefault("used", []).append(item["id"])

        meta = {
            "title": item["title"],
            "source": item["source"],
            "link": item["link"],
            "script": script,
            "video": str(video_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

        video_path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    save_state(state)

    print("Created videos:")

    for video in made:
        print(video)


if __name__ == "__main__":
    asyncio.run(main())
