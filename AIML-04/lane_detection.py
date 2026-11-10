"""
Autonomous Driving Perception Stack - Module 3: Lane Detection
Performs classical computer vision lane detection, curve fitting, curvature estimation,
and vehicle offset calculation on road images using OpenCV and NumPy.
"""

from pathlib import Path
import cv2
import numpy as np


def preprocess_frame(img: np.ndarray) -> np.ndarray:
    """
    Applies HLS color filtering (white and yellow) and Canny edge detection,
    followed by a Region of Interest (ROI) mask to extract lane line candidates.
    """
    h, w = img.shape[:2]

    # 1. HLS Color Space Thresholding
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_ch = hls[:, :, 0]
    l_ch = hls[:, :, 1]
    s_ch = hls[:, :, 2]

    # White lane mask (high lightness)
    white_mask = np.uint8(l_ch > 180) * 255

    # Yellow lane mask (hue between 15 and 35, high saturation and moderate lightness)
    yellow_mask = np.uint8((h_ch >= 15) & (h_ch <= 35) & (s_ch > 70) & (l_ch > 60)) * 255

    color_binary = cv2.bitwise_or(white_mask, yellow_mask)

    # 2. Gaussian Blur & Canny Edge Detection
    blur = cv2.GaussianBlur(l_ch, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    # Combine color and edge information
    combined_binary = cv2.bitwise_or(color_binary, canny)

    # 3. Region of Interest (ROI) Trapezoid Mask
    roi_mask = np.zeros_like(combined_binary)
    roi_corners = np.array([[
        (int(w * 0.0), h),
        (int(w * 0.10), int(h * 0.48)),
        (int(w * 0.90), int(h * 0.48)),
        (int(w * 1.0), h)
    ]], dtype=np.int32)
    cv2.fillPoly(roi_mask, roi_corners, 255)

    roi_binary = cv2.bitwise_and(combined_binary, roi_mask)
    return roi_binary


def get_perspective_matrices(h: int, w: int):
    """
    Computes Bird's Eye View transformation matrices (M and Minv)
    for Inverse Perspective Mapping (IPM).
    """
    src = np.float32([
        [w * 0.05, h],
        [w * 0.15, h * 0.50],
        [w * 0.85, h * 0.50],
        [w * 0.95, h]
    ])
    dst = np.float32([
        [w * 0.15, h],
        [w * 0.15, 0],
        [w * 0.85, 0],
        [w * 0.85, h]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def find_lane_pixels_sliding_window(warped_binary: np.ndarray):
    """
    Locates left and right lane boundary pixels using vertical histogram peaks
    and a multi-window sliding search.
    """
    h, w = warped_binary.shape[:2]

    # Histogram of bottom half of warped binary image
    histogram = np.sum(warped_binary[h // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(h // nwindows)
    margin = 60
    minpix = 30

    nonzero = warped_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def validate_lane_geometry(left_fit, right_fit, h: int, xm_per_pix: float = 3.7 / 700.0) -> tuple[bool, str]:
    """
    Validates that fitted left and right polynomial curves form a physically plausible lane corridor.
    Checks:
    1. Non-crossing constraint: x_right(y) > x_left(y) for all y.
    2. Plausible lane width bounds (between 2.2m and 5.2m in real-world metric space).
    """
    ploty = np.linspace(0, h - 1, 20)
    left_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Check 1: Non-crossing verification
    separations = right_x - left_x
    if np.any(separations <= 0):
        return False, "Validation Failed: Left and right lane boundaries cross each other."

    # Check 2: Plausible highway lane width (standard: ~3.7m)
    min_width_m = float(np.min(separations) * xm_per_pix)
    max_width_m = float(np.max(separations) * xm_per_pix)

    if min_width_m < 2.2:
        return False, f"Validation Failed: Lane width too narrow ({min_width_m:.2f} m < 2.2 m)."
    if max_width_m > 5.2:
        return False, f"Validation Failed: Lane width too wide ({max_width_m:.2f} m > 5.2 m)."

    return True, f"Geometry Validated (Corridor width: {min_width_m:.2f}m – {max_width_m:.2f}m)"


def fit_polynomial_and_curvature(leftx, lefty, rightx, righty, h, w):
    """
    Fits 2nd-order polynomials (x = Ay^2 + By + C) to left and right lane pixels,
    validates geometry, and computes estimated road curvature and vehicle lateral offset.
    """
    # Strict minimum pixel count check to reject weak/noisy fits
    if len(leftx) < 150 or len(rightx) < 150:
        print(f"[WARN] Insufficient lane pixels (Left={len(leftx)}, Right={len(rightx)} < 150 threshold).")
        return None, None, None, None

    # Fit 2nd-order polynomial in pixel space
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Conversions in x and y from pixels space to real-world meters (US/EU highway approximation)
    ym_per_pix = 30.0 / 720.0  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700.0   # meters per pixel in x dimension (standard 3.7m lane width)

    # Geometry validation (crossing and width bounds)
    is_valid, validation_msg = validate_lane_geometry(left_fit, right_fit, h, xm_per_pix)
    if not is_valid:
        print(f"[ERROR] {validation_msg}")
        return None, None, None, None
    print(f"[INFO] {validation_msg}")

    # Fit polynomials to real-world coordinates
    y_eval = h - 1
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Curvature calculation: R = ((1 + (2Ay + B)^2)^1.5) / |2A|
    left_curvature_m = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / (np.abs(2 * left_fit_cr[0]) + 1e-7)
    right_curvature_m = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / (np.abs(2 * right_fit_cr[0]) + 1e-7)
    estimated_curvature_m = float((left_curvature_m + right_curvature_m) / 2.0)

    # Vehicle Center Offset Calculation:
    # Position of left and right lane at bottom of the frame
    left_bottom_x = left_fit[0] * (h ** 2) + left_fit[1] * h + left_fit[2]
    right_bottom_x = right_fit[0] * (h ** 2) + right_fit[1] * h + right_fit[2]
    lane_center_x = (left_bottom_x + right_bottom_x) / 2.0
    car_center_x = w / 2.0

    # Positive = vehicle is right of center; Negative = vehicle is left of center
    estimated_offset_m = float((car_center_x - lane_center_x) * xm_per_pix)

    return left_fit, right_fit, estimated_curvature_m, estimated_offset_m


def render_lane_corridor(original_img, left_fit, right_fit, Minv, curvature_m, offset_m):
    """
    Draws the green drivable lane corridor and boundary curves on a top-down canvas,
    unwarps back to the driver perspective, and overlays perception telemetry.
    """
    h, w = original_img.shape[:2]
    ploty = np.linspace(0, h - 1, h)

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create top-down color canvas
    color_warp = np.zeros_like(original_img)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Fill green drivable corridor
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 220, 0))

    # Draw colored boundary lines
    cv2.polylines(color_warp, np.int_([pts_left]), isClosed=False, color=(255, 200, 0), thickness=16)
    cv2.polylines(color_warp, np.int_([pts_right]), isClosed=False, color=(0, 100, 255), thickness=16)

    # Unwarp top-down corridor back to driver camera perspective
    unwarped_lane = cv2.warpPerspective(color_warp, Minv, (w, h))

    # Blend with original camera frame
    annotated = cv2.addWeighted(original_img, 1.0, unwarped_lane, 0.40, 0)

    # Format telemetry strings with explicit (Estimate) labeling
    if curvature_m > 5000:
        curv_str = "Estimated Curvature: Straight (> 5000 m) [Estimate]"
    else:
        direction = "Left" if left_fit[0] < 0 else "Right"
        curv_str = f"Estimated Curvature: {curvature_m:.1f} m (Curve {direction}) [Estimate]"

    if abs(offset_m) < 0.03:
        offset_str = "Estimated Vehicle Offset: +0.00 m (Centered) [Estimate]"
    elif offset_m > 0:
        offset_str = f"Estimated Vehicle Offset: {offset_m:+.2f} m (Right of Center) [Estimate]"
    else:
        offset_str = f"Estimated Vehicle Offset: {offset_m:+.2f} m (Left of Center) [Estimate]"

    # Perception HUD Overlay
    cv2.rectangle(annotated, (10, 10), (620, 96), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (620, 96), (70, 70, 70), 2)

    cv2.putText(annotated, "MODULE 3: LANE PERCEPTION STACK", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(annotated, curv_str, (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(annotated, offset_str, (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)

    return annotated


def run_lane_detection(input_image_path: Path, output_image_path: Path):
    print("=" * 70)
    print("MODULE 3: CLASSICAL COMPUTER VISION LANE DETECTION")
    print("=" * 70)

    if not input_image_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    img = cv2.imread(str(input_image_path))
    if img is None:
        raise RuntimeError(f"cv2.imread failed to load: {input_image_path}")

    h, w = img.shape[:2]
    print(f"[INFO] Input Image: {input_image_path.name} | Dimensions: {w}x{h}")

    # 1. Preprocessing & Edge/Color Filtering
    roi_binary = preprocess_frame(img)

    # 2. Perspective Transform (Bird's Eye View)
    M, Minv = get_perspective_matrices(h, w)
    warped_binary = cv2.warpPerspective(roi_binary, M, (w, h), flags=cv2.INTER_LINEAR)

    # 3. Sliding Window Search
    leftx, lefty, rightx, righty = find_lane_pixels_sliding_window(warped_binary)
    print(f"[INFO] Detected Lane Pixels: Left={len(leftx)} | Right={len(rightx)}")

    # 4. Polynomial Fitting, Geometry Validation, & Curvature Estimation
    left_fit, right_fit, curvature_m, offset_m = fit_polynomial_and_curvature(
        leftx, lefty, rightx, righty, h, w
    )

    if left_fit is None or right_fit is None:
        print("[ERROR] Lane detection failed: weak fits or invalid lane geometry.")
        return None

    print(f"[INFO] Left Polynomial  (x = Ay^2 + By + C): {left_fit}")
    print(f"[INFO] Right Polynomial (x = Ay^2 + By + C): {right_fit}")
    print(f"[INFO] Estimated Road Curvature: {curvature_m:.1f} meters (Estimate)")
    print(f"[INFO] Estimated Vehicle Offset: {offset_m:+.3f} meters (Estimate)")

    # 5. Render Drivable Corridor & HUD Overlay
    annotated_output = render_lane_corridor(
        img, left_fit, right_fit, Minv, curvature_m, offset_m
    )

    # 6. Save Annotated Output Image & Validate Return Value
    save_success = cv2.imwrite(str(output_image_path), annotated_output)
    if not save_success:
        raise IOError(f"cv2.imwrite failed to save output image to: {output_image_path}")

    print("-" * 70)
    print(f"[SUCCESS] Output saved to: {output_image_path}")
    print(f"[SUCCESS] Output Dimensions: {annotated_output.shape[1]}x{annotated_output.shape[0]}")
    print("=" * 70)

    return {
        "left_pixels": len(leftx),
        "right_pixels": len(rightx),
        "left_fit": left_fit.tolist(),
        "right_fit": right_fit.tolist(),
        "curvature_m": curvature_m,
        "offset_m": offset_m,
        "dimensions": f"{annotated_output.shape[1]}x{annotated_output.shape[0]}",
        "output_path": str(output_image_path)
    }


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    in_img = project_dir / "sample_road.jpg"
    out_img = project_dir / "lane_detection_output.jpg"
    run_lane_detection(in_img, out_img)
