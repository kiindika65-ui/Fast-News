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
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps


# ============================================================
# SETTINGS
# ============================================================

PAGE_NAME = os.getenv("PAGE_NAME", "World Pulse Daily")
VOICE = os.getenv("VOICE", "en-US-GuyNeural")

VIDEO_SECONDS_MIN = int(os.getenv("VIDEO_SECONDS_MIN", "35"))
VIDEO_SECONDS_MAX = int(os.getenv("VIDEO_SECONDS_MAX", "55"))
MAX_OLD_ITEMS = int(os.getenv("MAX_OLD_ITEMS", "1000"))
MAX_VIDEOS_PER_RUN = int(os.getenv("MAX_VIDEOS_PER_RUN", "1"))

USE_RSS_IMAGES = os.getenv("USE_RSS_IMAGES", "true").lower() == "true"

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
ASSET_DIR = Path(os.getenv("ASSET_DIR", "assets"))
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
    "horoscope",
]


# ============================================================
# FOLDER FIX
# ============================================================

def prepare_folders():
    for folder in [OUTPUT_DIR, ASSET_DIR]:
        if folder.exists() and not folder.is_dir():
            folder.unlink()
        folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# STATE
# ============================================================

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"used": []}
    return {"used": []}


def save_state(state: dict):
    state["used"] = state.get("used", [])[-MAX_OLD_ITEMS:]
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"&nbsp;", " ", value)
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

    return clean_text(source)[:50]


def strip_google_source(title: str):
    if " - " in title:
        headline, source = title.rsplit(" - ", 1)
        if len(headline) > 12:
            return headline.strip(), source.strip()
    return title.strip(), ""


def item_id(title: str, link: str) -> str:
    base = re.sub(r"[^a-z0-9]+", " ", f"{title} {link}".lower()).strip()
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def is_good_item(title: str, summary: str) -> bool:
    title_lower = title.lower()

    if len(title) < 25 or len(title) > 190:
        return False

    if any(word in title_lower for word in BANNED_TITLE_WORDS):
        return False

    return True


# ============================================================
# GET IMAGE ONLY FROM RSS ITEM
# ============================================================

def extract_first_image_from_html(text: str) -> str:
    if not text:
        return ""

    match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', text, re.I)

    if match:
        return html.unescape(match.group(1))

    return ""


def get_rss_image_url(entry) -> str:
    """
    Only extracts image URL if the RSS item itself provides it.
    No external search.
    No stock site.
    """
    try:
        media_content = getattr(entry, "media_content", [])
        if media_content:
            for media in media_content:
                url = media.get("url", "")
                medium = media.get("medium", "")
                if url and ("image" in medium.lower() or url.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
                    return url
                if url:
                    return url
    except Exception:
        pass

    try:
        media_thumbnail = getattr(entry, "media_thumbnail", [])
        if media_thumbnail:
            for media in media_thumbnail:
                url = media.get("url", "")
                if url:
                    return url
    except Exception:
        pass

    try:
        links = getattr(entry, "links", [])
        for link in links:
            href = link.get("href", "")
            link_type = link.get("type", "")
            rel = link.get("rel", "")

            if href and (
                "image" in link_type.lower()
                or rel.lower() in ["enclosure", "thumbnail"]
                or href.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ):
                return href
    except Exception:
        pass

    raw_summary = getattr(entry, "summary", "")
    image = extract_first_image_from_html(raw_summary)

    if image:
        return image

    raw_content = ""

    try:
        if getattr(entry, "content", None):
            raw_content = entry.content[0].get("value", "")
    except Exception:
        raw_content = ""

    image = extract_first_image_from_html(raw_content)

    if image:
        return image

    return ""


def download_rss_image(image_url: str, item_id_value: str) -> Path | None:
    if not image_url:
        return None

    if not image_url.lower().startswith(("http://", "https://")):
        return None

    headers = {
        "User-Agent": "WorldPulseDailyBot/3.0"
    }

    try:
        response = requests.get(image_url, headers=headers, timeout=25)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        if "image" not in content_type and not image_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return None

        image_path = ASSET_DIR / f"rss_image_{item_id_value}.jpg"

        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(image_path, "JPEG", quality=92)

        return image_path

    except Exception as e:
        print(f"RSS image download failed: {e}")
        return None


# BytesIO import fallback placed here to keep imports clean
from io import BytesIO


# ============================================================
# RSS FETCH
# ============================================================

def fetch_items() -> list[dict]:
    items = []
    feeds = RSS_FEEDS[:]
    random.shuffle(feeds)

    headers = {
        "User-Agent": "WorldPulseDailyBot/3.0"
    }

    for feed_url in feeds:
        try:
            response = requests.get(feed_url, headers=headers, timeout=25)
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
        except Exception as e:
            print(f"Feed failed: {feed_url} -> {e}")
            continue

        for entry in parsed.entries[:25]:
            raw_title = clean_text(getattr(entry, "title", ""))
            summary = clean_text(getattr(entry, "summary", ""))
            link = clean_text(getattr(entry, "link", ""))
            source = source_from_entry(entry)

            title, title_source = strip_google_source(raw_title)

            if title_source:
                source = title_source

            if not is_good_item(title, summary):
                continue

            image_url = get_rss_image_url(entry)
            uid = item_id(title, link)

            items.append({
                "id": uid,
                "title": title,
                "summary": summary,
                "link": link,
                "source": source,
                "image_url": image_url,
                "feed": feed_url,
            })

    random.shuffle(items)
    return items


def pick_new_items(items: list[dict], state: dict, count: int) -> list[dict]:
    used = set(state.get("used", []))
    picked = []
    seen_titles = set()

    for item in items:
        title_key = re.sub(r"[^a-z0-9]+", " ", item["title"].lower()).strip()[:100]

        if item["id"] in used:
            continue

        if title_key in seen_titles:
            continue

        seen_titles.add(title_key)
        picked.append(item)

        if len(picked) >= count:
            break

    return picked


# ============================================================
# ORIGINAL NEWS SCRIPT
# ============================================================

def make_original_script(item: dict) -> str:
    title = item["title"].strip(" .")
    summary = item.get("summary", "").strip(" .")
    source = item.get("source", "RSS source")

    summary = clean_text(summary)

    if len(summary) > 210:
        summary = summary[:210].rsplit(" ", 1)[0] + "."

    openers = [
        "This is World Pulse Daily with a quick news update.",
        "Here is a fast update from the latest news feed.",
        "A new headline is now getting attention across the news cycle.",
        "This is a short news brief based on the latest RSS update.",
    ]

    middles = [
        f"The main headline is this: {title}.",
        f"Reports from {source} are focusing on this development: {title}.",
        f"The key update now being reported is: {title}.",
        f"According to the latest RSS item from {source}, the story centers on this: {title}.",
    ]

    context = ""

    if summary and summary.lower() not in title.lower():
        context = f" In simple words, {summary}"

    safety_line = (
        "Some details may change as more information becomes available, "
        "so viewers should check the original source for the newest update."
    )

    closers = [
        "We will continue watching for clearer and more reliable updates.",
        "This is a developing story, and verified details may change later.",
        "Follow trusted sources before sharing developing claims.",
    ]

    script = (
        f"{random.choice(openers)} "
        f"{random.choice(middles)}"
        f"{context} "
        f"{safety_line} "
        f"{random.choice(closers)}"
    )

    script = clean_text(script)
    words = script.split()

    if len(words) > 125:
        script = " ".join(words[:125]).rsplit(".", 1)[0] + "."

    return script


# ============================================================
# VOICE
# ============================================================

async def make_voice(text: str, out_mp3: Path):
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate="+2%",
        volume="+0%"
    )

    await communicate.save(str(out_mp3))


# ============================================================
# VISUAL HELPERS
# ============================================================

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

        if len(current) >= 7 or word.endswith((".", "?", "!")):
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


def make_generated_news_background(seed: int) -> Image.Image:
    random.seed(seed)

    base1 = np.array([5, 10, 24], dtype=np.uint8)
    base2 = np.array([20, 42, 85], dtype=np.uint8)

    y = np.linspace(0, 1, H)[:, None]
    grad = (base1 * (1 - y) + base2 * y).astype(np.uint8)
    img = np.repeat(grad[:, None, :], W, axis=1)

    im = Image.fromarray(img, "RGB").convert("RGBA")
    draw = ImageDraw.Draw(im)

    # Subtle map/grid style lines
    for x in range(-300, W + 300, 120):
        draw.line((x, 0, x + 400, H), fill=(255, 255, 255, 18), width=2)

    for y in range(0, H, 120):
        draw.line((0, y, W, y), fill=(255, 255, 255, 14), width=1)

    # Soft circles
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    for _ in range(18):
        x = random.randint(-250, W)
        y0 = random.randint(-250, H)
        r = random.randint(160, 460)
        alpha = random.randint(12, 42)
        d.ellipse((x, y0, x + r, y0 + r), fill=(255, 255, 255, alpha))

    overlay = overlay.filter(ImageFilter.GaussianBlur(30))

    im = Image.alpha_composite(im, overlay)
    return im.convert("RGB")


def cover_resize_image(img: Image.Image, width: int, height: int, zoom: float = 1.0) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")

    iw, ih = img.size
    target_ratio = width / height
    img_ratio = iw / ih

    if img_ratio > target_ratio:
        new_h = height
        new_w = int(height * img_ratio)
    else:
        new_w = width
        new_h = int(width / img_ratio)

    new_w = int(new_w * zoom)
    new_h = int(new_h * zoom)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - width) // 2
    top = (new_h - height) // 2

    return img.crop((left, top, left + width, top + height))


def make_image_background(image_path: Path | None, seed: int, t: float, duration: float) -> Image.Image:
    if image_path and image_path.exists():
        try:
            img = Image.open(image_path).convert("RGB")
            zoom = 1.04 + 0.04 * (t / max(duration, 0.1))
            bg = cover_resize_image(img, W, H, zoom=zoom)
            bg = bg.filter(ImageFilter.GaussianBlur(6))

            dark = Image.new("RGBA", (W, H), (0, 0, 0, 120))
            bg = Image.alpha_composite(bg.convert("RGBA"), dark).convert("RGB")
            return bg
        except Exception as e:
            print(f"Could not use RSS image: {e}")

    return make_generated_news_background(seed)


# ============================================================
# VIDEO CREATION
# ============================================================

def make_frame_builder(item: dict, script: str, duration: float, image_path: Path | None):
    title_font = get_font(56, True)
    caption_font = get_font(50, True)
    brand_font = get_font(38, True)
    label_font = get_font(34, True)
    small_font = get_font(30, False)
    tiny_font = get_font(26, False)

    title = item["title"]
    source = item.get("source", "RSS source")

    title_lines = wrap_lines(title, title_font, 900)[:4]

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

    def draw_gradient_overlay(im: Image.Image):
        im = im.convert("RGBA")

        top = Image.new("RGBA", (W, 420), (0, 0, 0, 0))
        top_px = top.load()

        for yy in range(420):
            alpha = int(200 * (1 - yy / 420))
            for xx in range(W):
                top_px[xx, yy] = (0, 0, 0, alpha)

        im.alpha_composite(top, (0, 0))

        bottom = Image.new("RGBA", (W, 760), (0, 0, 0, 0))
        bottom_px = bottom.load()

        for yy in range(760):
            alpha = int(230 * (yy / 760))
            for xx in range(W):
                bottom_px[xx, yy] = (0, 0, 0, alpha)

        im.alpha_composite(bottom, (0, H - 760))
        return im

    def frame(t: float):
        seed = int(item["id"][:6], 16)

        bg = make_image_background(image_path, seed, t, duration)
        im = draw_gradient_overlay(bg)
        draw = ImageDraw.Draw(im)

        progress = max(0, min(1, t / max(duration, 0.1)))

        # Top brand bar
        draw_round_rect(
            draw,
            (50, 55, 1030, 155),
            20,
            fill=(6, 12, 28, 230),
            outline=(255, 255, 255, 65),
            width=2
        )

        draw.text(
            (82, 82),
            PAGE_NAME.upper(),
            font=brand_font,
            fill=(255, 255, 255, 255)
        )

        draw_round_rect(
            draw,
            (765, 75, 1000, 135),
            14,
            fill=(185, 20, 32, 245)
        )

        draw.text(
            (798, 89),
            "RSS NEWS",
            font=tiny_font,
            fill=(255, 255, 255, 255)
        )

        # Top story label
        draw_round_rect(
            draw,
            (60, 895, 355, 970),
            14,
            fill=(185, 20, 32, 245)
        )

        draw.text(
            (90, 915),
            "TOP STORY",
            font=label_font,
            fill=(255, 255, 255, 255)
        )

        # Headline box
        draw_round_rect(
            draw,
            (50, 970, 1030, 1305),
            26,
            fill=(5, 12, 28, 232),
            outline=(255, 255, 255, 70),
            width=2
        )

        y = 1010

        for line in title_lines:
            draw.text(
                (85, y),
                line,
                font=title_font,
                fill=(255, 255, 255, 255)
            )
            y += 68

        draw.text(
            (85, 1250),
            f"Source: {source}"[:70],
            font=small_font,
            fill=(220, 230, 245, 235)
        )

        # Voice caption box
        cap = caption_at(t)
        cap_lines = wrap_lines(cap, caption_font, 900)[:3]

        draw_round_rect(
            draw,
            (50, 1370, 1030, 1665),
            26,
            fill=(255, 255, 255, 235),
            outline=(255, 255, 255, 80),
            width=2
        )

        y = 1415

        for line in cap_lines:
            draw.text(
                (85, y),
                line,
                font=caption_font,
                fill=(5, 12, 28, 255)
            )
            y += 64

        # Progress bar
        draw_round_rect(
            draw,
            (50, 1710, 1030, 1760),
            10,
            fill=(5, 12, 28, 225)
        )

        draw_round_rect(
            draw,
            (50, 1710, int(50 + 980 * progress), 1760),
            10,
            fill=(185, 20, 32, 245)
        )

        # Footer
        footer = "Original brief from RSS metadata • No stock websites • Verify developing stories"

        if USE_RSS_IMAGES and item.get("image_url"):
            footer = "Visual from RSS item if available • Source credited • Verify developing stories"

        draw.text(
            (65, 1810),
            footer[:85],
            font=tiny_font,
            fill=(245, 245, 245, 235)
        )

        return np.array(im.convert("RGB"))

    return frame


def safe_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:75] or "news_video"


def create_video(item: dict, script: str, audio_path: Path, image_path: Path | None) -> Path:
    audio = AudioFileClip(str(audio_path))

    duration = min(
        max(audio.duration + 0.6, VIDEO_SECONDS_MIN),
        VIDEO_SECONDS_MAX
    )

    frame_builder = make_frame_builder(item, script, duration, image_path)

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
        bitrate="4500k",
        threads=2,
        verbose=False,
        logger=None
    )

    clip.close()
    audio.close()

    return out_path


# ============================================================
# MAIN
# ============================================================

async def main():
    prepare_folders()

    state = load_state()

    items = fetch_items()
    picked = pick_new_items(items, state, MAX_VIDEOS_PER_RUN)

    if not picked:
        print("No fresh unused RSS item found.")
        print("Try adding more RSS feeds or clear state.json.")
        return

    made = []

    for item in picked:
        print("TITLE:", item["title"])
        print("SOURCE:", item["source"])
        print("LINK:", item["link"])
        print("RSS IMAGE:", item.get("image_url", ""))

        script = make_original_script(item)

        print("SCRIPT:", script)

        audio_path = OUTPUT_DIR / f"voice_{item['id']}.mp3"

        await make_voice(script, audio_path)

        image_path = None

        if USE_RSS_IMAGES and item.get("image_url"):
            image_path = download_rss_image(item["image_url"], item["id"])

        video_path = create_video(
            item=item,
            script=script,
            audio_path=audio_path,
            image_path=image_path
        )

        made.append(str(video_path))
        state.setdefault("used", []).append(item["id"])

        meta = {
            "title": item["title"],
            "source": item["source"],
            "link": item["link"],
            "rss_image_url": item.get("image_url", ""),
            "used_rss_image": str(image_path) if image_path else None,
            "script": script,
            "video": str(video_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Video generated only from RSS metadata and RSS-provided image if available.",
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
