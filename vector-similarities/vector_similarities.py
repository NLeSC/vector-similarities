import numba
import numpy


@numba.njit
def cosine_similarity_matrix(references: numpy.ndarray, queries: numpy.ndarray) -> numpy.ndarray:
    """Returns matrix of cosine similarity scores between all-vs-all vectors of
    references and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = numpy.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            scores[i, j] = cosine_similarity(references[i, :], queries[j, :])
    return scores


@numba.njit
def cosine_similarity(u: numpy.ndarray, v: numpy.ndarray) -> numpy.float64:
    """Calculate cosine similarity score.

    Parameters
    ----------
    u
        Input vector.
    v
        Input vector.

    Returns
    -------
    cosine_similarity
        The Cosine similarity score between vectors `u` and `v`.
    """
    assert u.shape[0] == v.shape[0], "Input vector must have same shape."
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cosine_score = 0
    if uu != 0 and vv != 0:
        cosine_score = uv / numpy.sqrt(uu * vv)
    return numpy.float64(cosine_score)
