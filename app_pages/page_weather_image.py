# app_pages/page_weather_image.py

import streamlit as st

from src.ui import inject_global_css
from src.models import predict_weather_severity_from_bytes


def app():
    """
    Streamlit page: classify mountain weather severity from an uploaded image.

    This page does NOT alter any of the main demand / cancellation pipelines.
    It is a supporting tool that shows how image data can be turned into a
    simple weather severity label (mild / moderate / severe).
    """
    inject_global_css()

    st.title("🖼️ Weather from Image")

    st.markdown(
        """
        <div class="card">
        <p>
        Upload a photo showing <strong>current mountain conditions</strong> and this
        prototype model will classify the overall <strong>weather severity</strong>.
        </p>
        <p>
        Behind the scenes, the app extracts simple, interpretable features from the
        image (brightness, colourfulness, proportion of bright pixels) and uses a
        small scikit-learn model trained on labelled examples.
        </p>
        <p>
        This is designed as a <strong>supporting tool</strong> – it does not replace
        professional judgement about mountain safety, but illustrates how image
        data can feed into the wider analytics workflow.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a mountain weather image",
        type=["jpg", "jpeg", "png", "webp"],
        help="For example: a photo of current conditions on a winter route.",
    )

    if uploaded_file is None:
        st.info("Choose an image file to see the predicted weather severity.")
        return

    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    if st.button("Classify weather severity"):
        try:
            # Read raw bytes for the model helper
            image_bytes = uploaded_file.read()

            result = predict_weather_severity_from_bytes(image_bytes)
            label = result["label"]
            proba = result.get("proba", {})

            # Accessible label explanation
            label_descriptions = {
                "mild": "Conditions look relatively clear or benign.",
                "moderate": "Some snow, cloud or reduced visibility is present.",
                "severe": "Conditions appear poor: heavy snow, storm or low visibility.",
            }
            description = label_descriptions.get(
                label,
                "Predicted conditions fall into this severity category.",
            )

            st.subheader("Predicted weather severity")
            st.markdown(
                f"**Label:** `{label}`  \n"
                f"{description}"
            )

            if proba:
                st.markdown("#### Class probabilities")
                # Convert to a small table for display
                st.table(
                    {
                        "class": list(proba.keys()),
                        "probability": [f"{p:.3f}" for p in proba.values()],
                    }
                )

            with st.expander("How this relates to the rest of the project"):
                st.markdown(
                    """
                    - The severity label can be mapped onto the same
                      **weather severity bins** used in the demand and
                      cancellation models (e.g. mild / moderate / severe).
                    - In a real deployment, this could help standardise how
                      guides record conditions or provide a quick visual
                      cross-check alongside forecast data.
                    """
                )

        except RuntimeError as e:
            # Model not available fallback
            st.error(
                "Weather severity model not available. "
                "Ensure `models/weather_severity_model.pkl` has been "
                "trained and committed."
            )
            st.caption(str(e))
        except Exception as e:  # pragma: no cover , defensive addition
            st.error("Something went wrong while classifying the image.")
            st.caption(str(e))
