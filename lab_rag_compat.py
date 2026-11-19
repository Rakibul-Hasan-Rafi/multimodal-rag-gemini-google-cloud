"""Gemini Gen AI SDK compatibility helpers for the uploaded Google Cloud multimodal RAG lab.
Learning-project adaptation for Gen AI SDK compatibility, not an official Google utility module.
Inspired by the official Google Cloud multimodal RAG sample; see NOTICE.md for attribution.
"""
from __future__ import annotations
import glob
import io
import mimetypes
import os
import time
from functools import lru_cache
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from IPython.display import display
from google.genai import types

_CLIENT = None
EMBED_MODEL = os.environ.get('EMBEDDING_MODEL', 'gemini-embedding-2')
DEFAULT_DIM = 1408


def set_global_client(client):
    global _CLIENT
    _CLIENT = client


def _client():
    if _CLIENT is None:
        raise RuntimeError('Run the Google Gen AI client initialization cell first.')
    return _CLIENT


def _retry(call, attempts=2):
    for i in range(attempts):
        try:
            return call()
        except Exception as exc:
            if '429' not in str(exc) or i == attempts - 1:
                raise
            print('429 quota/resource limit. Retrying once after 60 seconds...')
            time.sleep(60)


@lru_cache(maxsize=256)
def _text_embedding(text, dimensions):
    response = _retry(lambda: _client().models.embed_content(
        model=EMBED_MODEL,
        contents=text if text.strip() else ' ',
        config=types.EmbedContentConfig(output_dimensionality=int(dimensions)),
    ))
    return tuple(response.embeddings[0].values)


def get_text_embedding_from_text_embedding_model(text: str, return_array=False):
    v = _text_embedding(str(text), DEFAULT_DIM)
    return np.asarray(v, dtype=float) if return_array else list(v)


def get_image_embedding_from_multimodal_embedding_model(image_uri: str, embedding_size=512, text=None, return_array=False):
    path = Path(image_uri)
    mime_type = mimetypes.guess_type(str(path))[0] or 'image/png'
    parts = [types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)]
    if text:
        parts.append(types.Part.from_text(text=str(text)))
    response = _retry(lambda: _client().models.embed_content(
        model=EMBED_MODEL,
        contents=[types.Content(role='user', parts=parts)],
        config=types.EmbedContentConfig(output_dimensionality=int(embedding_size)),
    ))
    v = response.embeddings[0].values
    return np.asarray(v, dtype=float) if return_array else list(v)


def _part_from_image(img):
    """Treat long prompts as text, not as candidate filesystem paths."""
    if isinstance(img, PILImage.Image):
        out = io.BytesIO()
        img.convert('RGB').save(out, format='PNG')
        return types.Part.from_bytes(data=out.getvalue(), mime_type='image/png')
    if isinstance(img, (str, Path)):
        candidate = str(img)
        # Avoid OSError [Errno 36] for long multiline financial prompts.
        if len(candidate) < 180 and '\n' not in candidate and '\r' not in candidate:
            try:
                p = Path(candidate)
                if p.is_file():
                    return types.Part.from_bytes(
                        data=p.read_bytes(),
                        mime_type=mimetypes.guess_type(str(p))[0] or 'image/png',
                    )
            except (OSError, ValueError):
                pass
    return types.Part.from_text(text=str(img))


def get_gemini_response(generative_multimodal_model, model_input, stream=True, generation_config=None, safety_settings=None):
    inputs = model_input if isinstance(model_input, (list, tuple)) else [model_input]
    parts = [_part_from_image(x) for x in inputs]
    config = generation_config
    # Legacy 'safety_settings' dicts need not be translated for the financial sample.
    response = _retry(lambda: _client().models.generate_content(
        model=str(generative_multimodal_model),
        contents=[types.Content(role='user', parts=parts)],
        config=config,
    ))
    return response.text or ''


def get_document_metadata(generative_multimodal_model, pdf_folder_path, image_save_dir,
                          image_description_prompt, embedding_size=128,
                          generation_config=None, safety_settings=None,
                          add_sleep_after_page=False, sleep_time_after_page=2):
    global DEFAULT_DIM
    DEFAULT_DIM = int(embedding_size)
    Path(image_save_dir).mkdir(parents=True, exist_ok=True)
    text_rows, image_rows = [], []
    pdf_paths = sorted(glob.glob(os.path.join(pdf_folder_path, '*.pdf')))
    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)
        fname = Path(pdf_path).name
        print(f'Processing {fname} - {len(doc)} pages')
        for n, page in enumerate(doc, 1):
            text = page.get_text('text') or ''
            pieces = [text[i:i+1000] for i in range(0, len(text), 900)] or []
            if pieces:
                full_vec = get_text_embedding_from_text_embedding_model(text[:2000])
                for k, chunk in enumerate(pieces, 1):
                    text_rows.append(dict(file_name=fname, page_num=n, text=text,
                       text_embedding_page=full_vec, chunk_number=k,
                       chunk_text=chunk,
                       text_embedding_chunk=get_text_embedding_from_text_embedding_model(chunk)))
            seen = set()
            for ix, info in enumerate(page.get_images(full=True), 1):
                xref = info[0]
                if xref in seen:
                    continue
                seen.add(xref)
                try:
                    data = doc.extract_image(xref)
                    img = PILImage.open(io.BytesIO(data['image'])).convert('RGB')
                    if img.width < 120 or img.height < 60:
                        continue
                    target = Path(image_save_dir)/f'{Path(fname).stem}_image_{n}_{ix}_{xref}.png'
                    img.save(target)
                    desc = get_gemini_response(generative_multimodal_model,
                                               [image_description_prompt, img],
                                               generation_config=generation_config)
                    image_rows.append(dict(file_name=fname, page_num=n, img_num=ix,
                        img_path=str(target), img_desc=desc,
                        mm_embedding_from_img_only=get_image_embedding_from_multimodal_embedding_model(str(target), embedding_size),
                        text_embedding_from_image_description=get_text_embedding_from_text_embedding_model(desc)))
                except Exception as exc:
                    if '429' in str(exc) or '403' in str(exc):
                        raise
                    print(f'Skipped image on {fname} page {n}: {str(exc)[:100]}')
            if add_sleep_after_page:
                time.sleep(sleep_time_after_page)
        doc.close()
    text_df, image_df = pd.DataFrame(text_rows), pd.DataFrame(image_rows)
    print(f'Metadata: {len(text_df)} text chunks, {len(image_df)} images')
    return text_df, image_df


def _cosine(a, b):
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f'Embedding sizes differ ({a.size} vs {b.size}). Rebuild or reload compatible metadata.')
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a@b/d) if d else 0.0


def get_similar_text_from_query(query, text_metadata_df, column_name='', top_n=3, chunk_text=True, print_citation=False):
    if column_name not in text_metadata_df.columns:
        raise KeyError(column_name)
    dim = len(text_metadata_df.iloc[0][column_name])
    query_vec = _text_embedding(str(query), dim)
    scored = [(pos, _cosine(row[column_name], query_vec)) for pos, (_,row) in enumerate(text_metadata_df.iterrows())]
    scored.sort(key=lambda x:x[1], reverse=True)
    results = {}
    for i,(pos,score) in enumerate(scored[:top_n]):
        r=text_metadata_df.iloc[pos]
        results[i] = dict(file_name=r['file_name'],page_num=int(r['page_num']),cosine_score=round(score,4),
                          chunk_number=int(r.get('chunk_number',0)),chunk_text=r.get('chunk_text',''),
                          page_text=r.get('text',''))
    if print_citation:
        print_text_to_text_citation(results, chunk_text=chunk_text)
    return results


def get_similar_image_from_query(text_metadata_df, image_metadata_df, query='', image_query_path='',
                                 column_name='', image_emb=True, top_n=3, embedding_size=128):
    if column_name not in image_metadata_df.columns:
        raise KeyError(column_name)
    dim = len(image_metadata_df.iloc[0][column_name])
    if image_emb:
        query_vec = get_image_embedding_from_multimodal_embedding_model(image_query_path, embedding_size=dim)
    else:
        query_vec = _text_embedding(str(query), dim)
    scored=[]
    for pos, (_,r) in enumerate(image_metadata_df.iterrows()):
        if image_emb and os.path.abspath(str(r['img_path'])) == os.path.abspath(str(image_query_path)):
            continue
        scored.append((pos,_cosine(r[column_name],query_vec)))
    scored.sort(key=lambda x:x[1], reverse=True)
    results={}
    for i,(pos,score) in enumerate(scored[:top_n]):
        r=image_metadata_df.iloc[pos]
        matching_text = text_metadata_df[(text_metadata_df['file_name']==r['file_name']) &
                                         (text_metadata_df['page_num']==r['page_num'])]
        results[i]=dict(cosine_score=round(score,4),file_name=r['file_name'],img_path=r['img_path'],
             page_num=int(r['page_num']),image_object=PILImage.open(r['img_path']).copy(),
             image_description=r['img_desc'],page_text=matching_text.get('text',pd.Series([],dtype=str)).dropna().unique().tolist())
    return results


def print_text_to_text_citation(final_text, print_top=True, chunk_text=True):
    for i,r in final_text.items():
        print(f"Citation {i+1} | {r['file_name']} | Page {r['page_num']} | Score {r['cosine_score']}")
        print(('Chunk: '+str(r['chunk_number'])+'\n'+str(r['chunk_text'])) if chunk_text else str(r.get('page_text','')))
        if print_top:
            break


def print_text_to_image_citation(final_images, print_top=True):
    for i,r in final_images.items():
        print(f"Citation {i+1} | {r['file_name']} | Page {r['page_num']} | Score {r['cosine_score']}")
        print('Path:',r['img_path']);print('Image description:',r['image_description'])
        if print_top:
            break


def display_images(images, resize_ratio=.5):
    for img in images:
        im=PILImage.open(img) if isinstance(img,(str,Path)) else img
        if resize_ratio != 1:
            im=im.resize((max(1,int(im.width*resize_ratio)),max(1,int(im.height*resize_ratio))))
        display(im)
