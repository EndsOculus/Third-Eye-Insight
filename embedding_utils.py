import numpy as np


def compute_user_text_embeddings(chat_df, users, embedding_dim):
    """Compute normalized average embeddings for each user.

    Args:
        chat_df (pd.DataFrame): DataFrame containing 'sender_id' and 'text_embedding'.
        users (Iterable): Iterable of user identifiers.
        embedding_dim (int): Dimension of embedding vectors.

    Returns:
        dict: Mapping from user identifier to embedding vector.
    """
    user_text_embeddings = {}
    for user in users:
        embeds = chat_df[chat_df['sender_id'] == user]['text_embedding'].tolist()
        if embeds:
            avg_embed = np.mean(embeds, axis=0)
            norm_val = np.linalg.norm(avg_embed)
            user_text_embeddings[user] = avg_embed / norm_val if norm_val > 0 else avg_embed
        else:
            user_text_embeddings[user] = np.zeros(embedding_dim)
    return user_text_embeddings
