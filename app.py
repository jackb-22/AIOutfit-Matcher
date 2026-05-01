#!/usr/bin/env python3

import os
import streamlit as st
from main import run_pipeline

def get_cover_image(file_path):
    folder_path = os.path.dirname(file_path)
    try:
        valid_exts = ('.png', '.jpg', '.jpeg')
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        images.sort()

        if images:
            return os.path.join(folder_path, images[0])
    except Exception as e:
        pass

    return file_path

st.set_page_config(layout="wide")
st.title("Outfit Matcher")

uploaded = st.file_uploader("Upload an outfit image", type=["jpg", "jpeg", "png"])

if uploaded:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.read())

    with st.spinner("Analyzing outfit and searching database..."):
        results = run_pipeline("temp.jpg")

    for i, res_dict in enumerate(results):
        crop_img = res_dict["crop"]
        label = res_dict["label"]
        db_data = res_dict["db_results"]

        st.write("---")
        st.subheader(f"Detected Item {i+1} ({label.title()})")

        col_crop, col_matches = st.columns([1, 4])

        with col_crop:
            st.write("**What the AI saw:**")
            st.image(crop_img, use_container_width=True)

        with col_matches:
            st.write("**Top Database Matches:**")
            ids = db_data["ids"][0]
            metas = db_data["metadatas"][0]

            if not ids:
                st.info("No similar items found in the database.")
            else:
                match_cols = st.columns(5)
                for j in range(min(5, len(ids))):
                    with match_cols[j]:
                        best_image = get_cover_image(metas[j]["path"])

                        st.image(best_image, use_container_width=True)
                        st.caption(f"Cat: {metas[j].get('category', 'N/A')}")
