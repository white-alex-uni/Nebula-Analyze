import numpy as np

"""
Більше інформації: https://bottosson.github.io/posts/oklab/
"""

M1 = np.array([
        [0.8189330101, 0.3618667424, -0.1288597137],
        [0.0329845436, 0.9293118715, 0.0361456387],
        [0.0482003018, 0.2643662691, 0.6338517070],
], dtype=np.float32)

# Matrix M2 isolates lightness from the red/green and yellow/blue nerve paths
M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=np.float32)

# Inverse matrices for backward conversion
M1_inv = np.array([
    [ 4.0767416621, -3.3077115913,  0.2309699292],
    [-1.2684380046,  2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147,  1.7076147010]
], dtype=np.float32)

M2_inv = np.array([
    [1.0,  0.3963377774,  0.2158017574],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480]
], dtype=np.float32)

# Matrix sRGB (D65) -> XYZ
SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float32)

def rgb_2_oklab(rgb_input: np.ndarray) -> np.ndarray:
    """
    Перетворює масив пікселів RGB розміром [..., 3] у масив Oklab розміром [..., 3].
    Виконує гамма-декодування без використання сторонніх бібліотек.
    """
    input_arr = np.asarray(rgb_input)
    srgb_norm = input_arr.astype(np.float32)

    # Перевірка типу даних: цілочисельні масиви (напр. uint8) завжди нормалізуємо у [0.0, 1.0],
    # навіть якщо всі пікселі дуже темні (<= 1).
    if np.issubdtype(input_arr.dtype, np.integer) or np.any(srgb_norm > 1.0):
        srgb_norm = srgb_norm / 255.0

    srgb_norm = np.clip(srgb_norm, 0.0, 1.0)

    # Gamma linearization sRGB
    linear = np.where(
        srgb_norm <= 0.04045,
        srgb_norm / 12.92,
        ((srgb_norm + 0.055) / 1.055) ** 2.4
    )

    # OkLAB convert process: XYZ -> LMS -> Compression -> Oklab
    # Convert sRGB to CIE XYZ space (D65 white point reference)
    xyz = linear @ SRGB_TO_XYZ.T

    # Transform XYZ to LMS (Large, Medium, Small) Cone Response Space
    lms = xyz @ M1.T

    # Apply cube root compression
    lms_compressed = np.cbrt(lms)

    # Transform compressed LMS space to final Lab channels
    oklab = lms_compressed @ M2.T
    return oklab.astype(np.float32)

def oklab_2_rgb(oklab_input: np.ndarray | tuple | list, is_float: bool = False) -> np.ndarray:
    """
    Перетворює значення Oklab у колірний простір sRGB.
    Якщо is_float=True, повертає float32 у діапазоні [0.0, 1.0], інакше uint8 [0, 255].
    """
    oklab_arr = np.asarray(oklab_input, dtype=np.float32)

    # Oklab back to LMS
    lms_compressed = oklab_arr @ M2_inv.T

    # Undo compression (Cube it)
    lms = lms_compressed ** 3

    # LMS back to Linear RGB
    linear = lms @ M1_inv.T

    # Запобігаємо NaN при піднесенні від'ємних значень (out-of-gamut) до дробового степеня 1/2.4
    linear_clipped = np.clip(linear, 0.0, 1.0)

    # Re-apply Gamma Curve (Gamma encoding)
    srgb_gamma = np.where(
        linear_clipped <= 0.0031308,
        linear_clipped * 12.92,
        1.055 * (np.maximum(linear_clipped, 0.0031308) ** (1.0 / 2.4)) - 0.055
    )

    # Clip out-of-gamut values safely to 0-1
    srgb_float = np.clip(srgb_gamma, 0.0, 1.0).astype(np.float32)

    if is_float:
        return srgb_float

    return np.rint(srgb_float * 255.0).astype(np.uint8)