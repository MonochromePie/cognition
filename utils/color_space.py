import numpy as np

# Making all colors perceptually uniform

# Transformation matrices
_RGB_TO_XYZ = np.array([
    [0.41239079926595934, 0.357584339383878,   0.1804807884018343],
    [0.21263900587151027, 0.715168678767756,   0.07219231536073371],
    [0.01933081871559182, 0.11919477979462598, 0.9505321522496607]
])

_XYZ_TO_RGB = np.array([
    [ 3.2409699419045226,  -1.537383177570094,   -0.4986107602930034],
    [-0.9692436362808796,   1.8759675015077202,   0.04155505740717559],
    [ 0.05563007969699366, -0.20397695888897652,  1.0569715142428786]
])

_XYZ_TO_LMS = np.array([
    [0.8190224379967030, 0.3619062600528904, -0.1288737815209879],
    [0.0329836539323885, 0.9292868615863434,  0.0361446663506424],
    [0.0481771893596242, 0.2642395317527308,  0.6335478284694309]
])

_LMS_TO_XYZ = np.array([
    [ 1.2268798758459243, -0.5578149944602171,  0.2813910456659647],
    [-0.0405757452148008,  1.1122868032803170, -0.0717110580655164],
    [-0.0763729366746601, -0.4214933324022432,  1.5869240198367816]
])

_LMSg_TO_LAB = np.array([
    [0.2104542683093140,  0.7936177747023054, -0.0040720430116193],
    [1.9779985324311684, -2.4285922420485799,  0.4505937096174110],
    [0.0259040424655478,  0.7827717124575296, -0.8086757549230774]
])

_LAB_TO_LMSg = np.array([
    [1,  0.3963377773761749,  0.2158037573099136],
    [1, -0.1055613458156586, -0.0638541728258133],
    [1, -0.0894841775298119, -1.2914855480194092]
])


def _rgb_to_srgb_linear(rgb):
    """Convert sRGB to linear RGB."""
    abs_rgb = np.abs(rgb)
    sign = np.sign(rgb)
    return np.where(
        abs_rgb <= 0.04045,
        rgb / 12.92,
        sign * (((abs_rgb + 0.055) / 1.055) ** 2.4)
    )


def _srgb_linear_to_rgb(rgb):
    """Convert linear RGB to sRGB."""
    abs_rgb = np.abs(rgb)
    sign = np.sign(rgb)
    return np.where(
        abs_rgb > 0.0031308,
        sign * (1.055 * (abs_rgb ** (1 / 2.4)) - 0.055),
        12.92 * rgb
    )


def _apply_matrix(image, matrix):
    """Apply 3x3 transformation matrix to image with shape (H, W, 3)."""
    return np.tensordot(image, matrix.T, axes=([-1], [0]))


def _oklab_to_oklch(lab):
    """Convert OKLab to OKLCH."""
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    c = np.sqrt(a ** 2 + b ** 2)
    h = np.degrees(np.arctan2(b, a)) % 360
    # Set hue to NaN for achromatic colors
    h = np.where((np.abs(a) < 0.0002) & (np.abs(b) < 0.0002), np.nan, h)
    return np.stack([l, c, h], axis=-1)


def _oklch_to_oklab(lch):
    """Convert OKLCH to OKLab."""
    l, c, h = lch[..., 0], lch[..., 1], lch[..., 2]
    h_rad = np.radians(h)
    a = np.where(np.isnan(h), 0, c * np.cos(h_rad))
    b = np.where(np.isnan(h), 0, c * np.sin(h_rad))
    return np.stack([l, a, b], axis=-1)


def rgb_to_oklch(image):
    """Convert RGB image to OKLCH color space.
    
    Args:
        image: numpy array of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        OKLCH image with L in [0, 1], C >= 0, H in [0, 360) or NaN for grays
    """
    if image.dtype == 'uint8':
        print('uint8')
        image = image.astype(np.float64) / 255
    linear = _rgb_to_srgb_linear(image)
    xyz = _apply_matrix(linear, _RGB_TO_XYZ)
    lms = _apply_matrix(xyz, _XYZ_TO_LMS)
    lms_g = np.cbrt(lms)
    lab = _apply_matrix(lms_g, _LMSg_TO_LAB)
    return _oklab_to_oklch(lab)


def oklch_to_rgb(image, clip=True):
    """Convert OKLCH image to RGB color space.
    
    Args:
        image: numpy array of shape (H, W, 3) with L, C, H values
        clip: if True, clamp output to [0, 1]
    
    Returns:
        RGB image with values in [0, 1] (if clipped)
    """
    lab = _oklch_to_oklab(image)
    lms_g = _apply_matrix(lab, _LAB_TO_LMSg)
    lms = lms_g ** 3
    xyz = _apply_matrix(lms, _LMS_TO_XYZ)
    linear = _apply_matrix(xyz, _XYZ_TO_RGB)
    rgb = _srgb_linear_to_rgb(linear)
    if clip:
        rgb = np.clip(rgb, 0, 1)
    print(rgb.dtype)
    if image.dtype == 'float64':
        rgb = rgb * 255
    return rgb.astype(np.uint8)
