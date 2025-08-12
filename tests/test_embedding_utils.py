import numpy as np
import pandas as pd
from embedding_utils import compute_user_text_embeddings


def test_compute_user_text_embeddings_missing_user():
    sample_embed = np.array([1.0, 2.0, 2.0])
    chat_df = pd.DataFrame({
        'sender_id': ['u1'],
        'text_embedding': [sample_embed]
    })
    users = ['u1', 'u2']
    result = compute_user_text_embeddings(chat_df, users, embedding_dim=len(sample_embed))
    assert 'u2' in result
    assert np.array_equal(result['u2'], np.zeros(len(sample_embed)))
    assert np.isclose(np.linalg.norm(result['u1']), 1.0)
