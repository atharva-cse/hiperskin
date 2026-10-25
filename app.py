
import os
import html

import cv2
import numpy as np
import streamlit as st

from PIL import Image, ImageOps
from dotenv import load_dotenv
from google import genai
from streamlit_image_comparison import image_comparison


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HiperSkin | AI Image Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()

# Use the API key explicitly to avoid environment-variable ambiguity.
API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)


# --------------------------------------------------
# PROFESSIONAL DARK UI
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --background: #080d18;
        --surface: #111a2b;
        --surface2: #151f33;
        --border: rgba(148, 163, 184, 0.16);
        --text: #f1f5f9;
        --muted: #94a3b8;
        --teal: #2dd4bf;
        --purple: #a78bfa;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 8% 0%,
                rgba(45, 212, 191, 0.09),
                transparent 34%
            ),
            radial-gradient(
                ellipse at 95% 8%,
                rgba(139, 92, 246, 0.10),
                transparent 30%
            ),
            var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 24, 0.8);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1280px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label {
        color: var(--text);
    }

    .hero {
        padding: 30px 32px;
        margin: 6px 0 24px 0;
        border: 1px solid var(--border);
        border-radius: 24px;
        background: linear-gradient(
            125deg,
            rgba(17, 26, 43, 0.97),
            rgba(20, 30, 52, 0.88)
        );
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        right: -50px;
        top: -75px;
        border-radius: 50%;
        background: rgba(45, 212, 191, 0.09);
        filter: blur(8px);
    }

    .eyebrow {
        display: inline-block;
        color: #5eead4;
        background: rgba(45, 212, 191, 0.10);
        border: 1px solid rgba(45, 212, 191, 0.23);
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 15px;
    }

    .hero-title {
        font-size: clamp(32px, 5vw, 48px);
        font-weight: 800;
        letter-spacing: -1.8px;
        margin: 0 0 10px 0;
        line-height: 1.12;
        color: #f8fafc;
    }

    .gradient-text {
        background: linear-gradient(
            90deg,
            #2dd4bf,
            #60a5fa,
            #c4b5fd
        );
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-description {
        color: #a8b5c9;
        font-size: 15px;
        line-height: 1.8;
        max-width: 720px;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 20px;
        font-weight: 750;
        margin: 25px 0 5px 0;
        color: #f8fafc;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .panel {
        background: rgba(17, 26, 43, 0.88);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .panel-title {
        font-size: 16px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    .panel-caption {
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 15px;
    }

    .metric-card {
        background: linear-gradient(
            145deg,
            rgba(21, 31, 51, 0.98),
            rgba(14, 23, 39, 0.98)
        );
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 19px;
        min-height: 116px;
    }

    .metric-label {
        font-size: 11px;
        color: #94a3b8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 800;
        color: #f8fafc;
    }

    .metric-note {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 5px;
    }

    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 11px;
        font-weight: 700;
        background: rgba(45, 212, 191, 0.10);
        color: #5eead4;
        border: 1px solid rgba(45, 212, 191, 0.20);
    }

    .ai-card {
        background: linear-gradient(
            130deg,
            rgba(45, 212, 191, 0.07),
            rgba(167, 139, 250, 0.09)
        );
        border: 1px solid rgba(167, 139, 250, 0.25);
        border-radius: 18px;
        padding: 22px;
        margin-top: 10px;
    }

    .ai-title {
        font-size: 17px;
        font-weight: 750;
        color: #e9d5ff;
        margin-bottom: 12px;
    }

    .ai-body {
        color: #d2dbea;
        font-size: 14px;
        line-height: 1.9;
        white-space: pre-wrap;
    }

    .stButton > button {
        width: 100%;
        min-height: 47px;
        border-radius: 12px;
        border: 1px solid rgba(45, 212, 191, 0.35);
        background: linear-gradient(100deg, #0f766e, #0d9488);
        color: white;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #5eead4;
        background: linear-gradient(100deg, #115e59, #0f766e);
        color: white;
    }

    [data-testid="stFileUploader"] {
        background: rgba(8, 13, 24, 0.45);
        border: 1px dashed rgba(148, 163, 184, 0.35);
        border-radius: 14px;
        padding: 8px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
    }

    [data-testid="stFileUploader"] section {
        color: #cbd5e1;
    }

    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.14);
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding: 28px 0 8px 0;
    }

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 2rem 1rem;
        }

        .hero {
            padding: 22px 19px;
        }

        .metric-card {
            padding: 14px;
            min-height: 100px;
        }

        .metric-value {
            font-size: 22px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

IMAGE_SIZE = (512, 512)


def prepare_image(uploaded_file):
    """Open, orient and resize an uploaded image."""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.fit(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
    )
    return image


def align_images(before_image, after_image):
    """
    Try to align the After image to the Before image with ORB.
    Return the original After image if alignment is unreliable.
    """
    before = cv2.cvtColor(
        np.array(before_image),
        cv2.COLOR_RGB2GRAY,
    )
    after = cv2.cvtColor(
        np.array(after_image),
        cv2.COLOR_RGB2GRAY,
    )

    orb = cv2.ORB_create(nfeatures=2000)

    kp1, des1 = orb.detectAndCompute(before, None)
    kp2, des2 = orb.detectAndCompute(after, None)

    if des1 is None or des2 is None:
        return after_image, False

    if len(kp1) < 10 or len(kp2) < 10:
        return after_image, False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des2, des1, k=2)

    good_matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]

    if len(good_matches) < 12:
        return after_image, False

    source_points = np.float32(
        [kp2[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    destination_points = np.float32(
        [kp1[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    matrix, inliers = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        5.0,
    )

    if matrix is None or inliers is None:
        return after_image, False

    if int(inliers.sum()) < 10:
        return after_image, False

    height, width = before.shape

    aligned = cv2.warpPerspective(
        np.array(after_image),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return Image.fromarray(aligned), True


def compare_images(before_image, after_image):
    """
    Calculate changed pixels and produce a red-highlight overlay.
    A pixel is considered changed when its largest RGB-channel
    difference exceeds 30.
    """
    before = np.asarray(before_image, dtype=np.int16)
    after = np.asarray(after_image, dtype=np.int16)

    difference = np.abs(before - after)

    # Ignore very small channel variations.
    changed_mask = np.max(difference, axis=2) > 30

    difference_percentage = (
        float(np.count_nonzero(changed_mask))
        / changed_mask.size
    ) * 100.0

    # Build a red overlay over the Before image.
    overlay = np.asarray(before_image).copy()

    red = np.zeros_like(overlay)
    red[:, :, 0] = 255

    blended = (
        overlay.astype(np.float32) * 0.48
        + red.astype(np.float32) * 0.52
    ).astype(np.uint8)

    overlay[changed_mask] = blended[changed_mask]

    # Difference map: brighter pixels indicate stronger differences.
    difference_map = np.clip(
        difference * 3,
        0,
        255,
    ).astype(np.uint8)

    return (
        difference_percentage,
        Image.fromarray(overlay),
        Image.fromarray(difference_map),
        changed_mask,
    )


# --------------------------------------------------
# GEMINI AI SUMMARY
# --------------------------------------------------

def generate_ai_summary(difference_percentage):
    """
    Send the calculated numeric percentage to Gemini.
    The uploaded images are not sent to the AI.
    """
    if not API_KEY:
        return (
            "Gemini AI summary is unavailable because no API key "
            "was found. Add GEMINI_API_KEY to your .env file."
        )

    prompt = f"""
You are an image comparison assistant.

The image comparison program calculated a changed-pixel percentage
of {difference_percentage:.2f}%.

Write a short, clear report in English with:
1. The calculated percentage.
2. A cautious interpretation of the amount of pixel difference.
3. A note that pixel difference can be caused by lighting, camera
   angle, alignment, compression or actual changes.
4. A reminder that this number alone does not identify the cause
   or prove a medical diagnosis.

Do not claim to have seen either image. You receive only the
numeric result, not the images.

Keep the response under 120 words.
"""

    try:
        client = genai.Client(api_key=API_KEY)

        # Try the model that worked in the existing project first.
        models = [
            "gemini-3.8-flash",
            "gemini-3.5-flash-lite",
        ]

        errors = []

        for model_name in models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )

                answer = getattr(response, "text", None)

                if answer and answer.strip():
                    return answer.strip()

                errors.append(f"{model_name}: empty response")

            except Exception as error:
                errors.append(f"{model_name}: {error}")

        return (
            "The image comparison completed, but Gemini could not "
            "generate a summary. Check the API key, model availability "
            "and API quota. Details: "
            + " | ".join(errors)
        )

    except Exception as error:
        return f"Gemini AI could not be reached: {error}"


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-POWERED IMAGE ANALYSIS</div>
        <div class="hero-title">
            Hiper<span class="gradient-text">Skin</span>
        </div>
        <p class="hero-description">
            Compare two images, detect pixel-level differences,
            visualize changed areas and generate an AI-assisted
            comparison report — all in one workspace.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Small feature labels.
tag1, tag2, tag3 = st.columns(3)

with tag1:
    st.markdown(
        '<div class="status-pill">✦ Before / After Slider</div>',
        unsafe_allow_html=True,
    )

with tag2:
    st.markdown(
        '<div class="status-pill">✦ OpenCV Image Analysis</div>',
        unsafe_allow_html=True,
    )

with tag3:
    st.markdown(
        '<div class="status-pill">✦ Gemini AI Summary</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# UPLOAD SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">01 &nbsp; Image workspace</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    'Upload both images to start the comparison.'
    '</div>',
    unsafe_allow_html=True,
)

before_col, after_col = st.columns(2, gap="large")

with before_col:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">🟢 Before image</div>
            <div class="panel-caption">
                Original or reference image
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    before_file = st.file_uploader(
        "Choose Before image",
        type=["jpg", "jpeg", "png"],
        key="before_image_upload",
    )

with after_col:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">🟣 After image</div>
            <div class="panel-caption">
                New image to compare against the reference
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    after_file = st.file_uploader(
        "Choose After image",
        type=["jpg", "jpeg", "png"],
        key="after_image_upload",
    )


# --------------------------------------------------
# MAIN COMPARISON
# --------------------------------------------------

if before_file is None or after_file is None:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Ready when you are</div>
            <div class="panel-caption">
                Add both images above. HiperSkin will resize them,
                attempt alignment and calculate their pixel differences.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    try:
        before_image = prepare_image(before_file)
        after_image_original = prepare_image(after_file)

        # Keep uploads in memory; no images are written to disk.
        with st.spinner("Preparing images and checking alignment..."):
            after_image, aligned = align_images(
                before_image,
                after_image_original,
            )

        if aligned:
            st.success(
                "Automatic alignment completed. The images were "
                "aligned using detected visual features."
            )
        else:
            st.warning(
                "Automatic alignment was not reliable. The original "
                "resized images will be compared."
            )

        (
            difference_percentage,
            overlay_image,
            difference_map,
            changed_mask,
        ) = compare_images(before_image, after_image)

        st.markdown(
            '<div class="section-heading">'
            '02 &nbsp; Interactive comparison'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Move the slider to reveal either image.'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            image_comparison(
                img1=before_image,
                img2=after_image,
                label1="BEFORE",
                label2="AFTER",
                width=900,
                starting_position=50,
                show_labels=True,
            )

        # Metrics
        st.markdown(
            '<div class="section-heading">'
            '03 &nbsp; Comparison insights'
            '</div>',
            unsafe_allow_html=True,
        )

        metric1, metric2, metric3 = st.columns(3, gap="medium")

        with metric1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Pixel difference</div>
                    <div class="metric-value">
                        {difference_percentage:.2f}%
                    </div>
                    <div class="metric-note">
                        Pixels exceeding the difference threshold
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric2:
            changed_pixels = int(np.count_nonzero(changed_mask))
            total_pixels = int(changed_mask.size)

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Changed pixels</div>
                    <div class="metric-value">
                        {changed_pixels:,}
                    </div>
                    <div class="metric-note">
                        Out of {total_pixels:,} compared pixels
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric3:
            alignment_status = "Aligned" if aligned else "Fallback"

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Alignment</div>
                    <div class="metric-value" style="font-size:23px">
                        {alignment_status}
                    </div>
                    <div class="metric-note">
                        OpenCV ORB feature matching
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Visual explanations
        st.markdown(
            '<div class="section-heading">'
            '04 &nbsp; Difference visualization'
            '</div>',
            unsafe_allow_html=True,
        )

        view1, view2 = st.columns(2, gap="large")

        with view1:
            st.markdown(
                '<div class="panel-title">🔴 Changed-area overlay</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="panel-caption">'
                'Red regions mark pixels that crossed the threshold.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.image(
                overlay_image,
                caption="Changed regions highlighted in red",
                width="stretch",
            )

        with view2:
            st.markdown(
                '<div class="panel-title">◻ Difference map</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="panel-caption">'
                'Brighter regions indicate larger pixel differences.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.image(
                difference_map,
                caption="Amplified RGB difference map",
                width="stretch",
            )

        # Downloadable visual outputs
        st.markdown(
            '<div class="section-heading">'
            '05 &nbsp; Export results'
            '</div>',
            unsafe_allow_html=True,
        )

        export1, export2 = st.columns(2, gap="medium")

        with export1:
            import io

            overlay_buffer = io.BytesIO()
            overlay_image.save(overlay_buffer, format="PNG")

            st.download_button(
                "⬇ Download red overlay (PNG)",
                data=overlay_buffer.getvalue(),
                file_name="hiperskin_red_overlay.png",
                mime="image/png",
                use_container_width=True,
            )

        with export2:
            diff_buffer = io.BytesIO()
            difference_map.save(diff_buffer, format="PNG")

            st.download_button(
                "⬇ Download difference map (PNG)",
                data=diff_buffer.getvalue(),
                file_name="hiperskin_difference_map.png",
                mime="image/png",
                use_container_width=True,
            )

        # AI report
        st.markdown(
            '<div class="section-heading">'
            '06 &nbsp; AI analysis'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-subtitle">
                Generate a written interpretation of the calculated
                difference percentage.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "✦ Generate Gemini AI summary",
            key="generate_ai_summary",
        ):
            with st.spinner("Generating your AI summary..."):
                summary = generate_ai_summary(difference_percentage)

            st.session_state["hiperskin_ai_summary"] = summary
            st.session_state["hiperskin_summary_key"] = (
                f"{before_file.name}|{after_file.name}|"
                f"{difference_percentage:.4f}"
            )

        current_summary_key = (
            f"{before_file.name}|{after_file.name}|"
            f"{difference_percentage:.4f}"
        )

        if (
            st.session_state.get("hiperskin_summary_key")
            == current_summary_key
            and st.session_state.get("hiperskin_ai_summary")
        ):
            summary = st.session_state["hiperskin_ai_summary"]

            st.markdown(
                '<div class="ai-card">'
                '<div class="ai-title">'
                '✦ HiperSkin AI report'
                '</div>'
                f'<div class="ai-body">{html.escape(summary)}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            st.download_button(
                "⬇ Download AI summary (TXT)",
                data=summary,
                file_name="hiperskin_ai_summary.txt",
                mime="text/plain",
            )

        st.caption(
            "Privacy: the comparison is performed locally. The Gemini "
            "summary receives the calculated percentage, not the images. "
            "Image differences alone cannot establish a medical diagnosis."
        )

    except Exception as error:
        st.error(
            "Something went wrong while processing the images. "
            "Check the error details below."
        )
        with st.expander("Technical error details"):
            st.code(str(error))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <hr>
    <div class="footer">
        HiperSkin · AI-assisted image comparison<br>
        Built with Python, Streamlit, OpenCV and Gemini
    </div>
    """,
    unsafe_allow_html=True,
)
