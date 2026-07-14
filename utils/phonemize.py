import re


def g2p_resolve(word, g2p_model):
    """Call G2P to generate pronunciation (used for handling OOV words)."""
    try:
        result = g2p_model.rewriter(word.lower())
        if result and result[0][0]:
            return result[0][0].split()
    except Exception:
        return None
    return None


def text_norm(s):
    """
    Text normalization (keep internal apostrophes like don't, it's; remove quote-like apostrophes and other punctuation):
    1. Lowercase the text
    2. Keep apostrophes between letters (e.g. don't)
    3. Remove apostrophes that are not between letters (used as quotes or standalone)
    4. Remove other common punctuation marks (.,;!?()[]-"“” etc.)
    5. Collapse multiple spaces into a single space
    """
    s = s.lower()

    # First temporarily replace apostrophes between letters (a'b) with a placeholder to avoid deletion
    # Support both ASCII ' and Unicode ’, ‘
    APOST = "<<<APOST>>>"  # Placeholder string (ensured not to appear in normal sentences)
    s = re.sub(r"(?<=[A-Za-z0-9])['\u2019\u2018](?=[A-Za-z0-9])", APOST, s)

    # Remove all remaining apostrophes (these are quotes or isolated marks)
    s = re.sub(r"['\u2019\u2018]", " ", s)

    # Remove other punctuation (while keeping internal apostrophes protected by the placeholder)
    s = re.sub(r"[,\.\!\?\;\:\(\)\[\]\"“”\-]", " ", s)

    # Restore internal apostrophes back to ASCII apostrophe (or to the original character if needed)
    s = s.replace(APOST, "'")

    # Merge extra spaces
    s = " ".join(s.split())

    return s


# ---------------- Core conversion ----------------
def sentence_to_phones(sentence, word2phones, g2p_model):
    """
    Convert sentence to phones:
    1. Split the original sentence and keep punctuation positions to insert sil later
    2. Insert <eps> at punctuation positions
    3. Add <eps> at the beginning and end of the sentence
    """
    original_sentence = sentence  # Save the original sentence
    sentence = text_norm(sentence)

    phone_sequence = ["<eps>"]  # Initial silence
    oov_list = []

    # Split the original sentence to locate punctuation positions

    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;!?]", original_sentence)

    for token in tokens:
        if re.match(r"[.,;!?]", token):  # Punctuation
            phone_sequence.append("<eps>")
        else:
            word = text_norm(token)  # Normalize word

            if word not in word2phones:
                g2p_ph = g2p_resolve(word, g2p_model)
                if g2p_ph:
                    phone_sequence.extend(g2p_ph)
                else:
                    phone_sequence.append(
                        "spn"
                    )  # If it really cannot be handled, use a short pause

                oov_list.append(word)

            else:
                pron, _ = max(word2phones[word].items(), key=lambda x: x[1])
                phone_sequence.extend(pron.split())

    if phone_sequence[-1] != '<eps>':
        phone_sequence.append("<eps>")  # Ending silence
    return phone_sequence, oov_list
