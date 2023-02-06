# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining a color-based blob detector for camera images."""

from typing import Mapping, Optional, Tuple

from absl import logging
import cv2
from dmr_vision import detector
from dmr_vision import types
import numpy as np


class BlobDetector(detector.ImageDetector):
  """Color-based blob detector."""

  def __init__(self,
               color_ranges: Mapping[str, types.ColorRange],
               scale: float = (1. / 6.),
               min_area: int = 230,
               mask_points: Optional[types.MaskPoints] = None,
               visualize: bool = False,
               toolkit: bool = False):
    """Constructs a `BlobDetector` instance.

    Args:
      color_ranges: A mapping between a given blob name and the range of YUV
        color used to segment it from an image.
      scale: Rescaling image factor. Used for increasing the frame rate, at the
        cost of reducing the precision of the blob barycenter and controur.
      min_area: The minimum area the detected blob must have.
      mask_points: (u, v) coordinates defining a closed regions of interest in
        the image where the blob detector will not look for blobs.
      visualize: Whether to output a visualization of the detected blob or not.
      toolkit: Whether to display a YUV GUI toolkit for parameter tuning.
        Enabling this implcitly sets `visualize = True`.
    """
    self._color_ranges = color_ranges
    self._scale = np.array(scale)
    self._min_area = min_area
    self._mask_points = mask_points if mask_points is not None else ()
    self._visualize = visualize
    self._mask = None
    self._toolkit = toolkit

    if self._toolkit:
      self._visualize = True

      self._window_name = "UV toolkit"
      self._window_size = (800, 1000)
      cv2.namedWindow(
          self._window_name,
          cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
      cv2.resizeWindow(self._window_name, self._window_size)

      self._trackbar_scale = 1000
      num_colors = len(self._color_ranges.keys())
      if num_colors > 1:
        cv2.createTrackbar("Color selector", self._window_name, 0,
                           len(self._color_ranges.keys()) - 1,
                           self._callback_change_color)

      cv2.createTrackbar("Subsampling", self._window_name, 5, 10,
                         lambda x: None)
      cv2.setTrackbarMin("Subsampling", self._window_name, 1)

      self._u_range_trackbar = CreateRangeTrackbar(self._window_name, "U min",
                                                   "U max", self._color_ranges,
                                                   "U", self._trackbar_scale)

      self._v_range_trackbar = CreateRangeTrackbar(self._window_name, "V min",
                                                   "V max", self._color_ranges,
                                                   "V", self._trackbar_scale)

      self._callback_change_color(0)

  def __del__(self):
    if self._toolkit:
      cv2.destroyAllWindows()

  def __call__(self,
               image: np.ndarray) -> Tuple[types.Centers, types.Detections]:
    """Finds color blobs in the image.

    Args:
      image: the input image.

    Returns:
      A dictionary mapping a blob name with
       - the (u, v) coordinate of its barycenter, if found;
       - `None`, otherwise;
      and a dictionary mapping a blob name with
       - its contour superimposed on the input image;
       - `None`, if `BlobDetector` is run with `visualize == False`.
    """
    # Preprocess the image.
    image = self._preprocess(image)
    # Convert the image to YUV.
    yuv_image = cv2.cvtColor(image.astype(np.float32) / 255., cv2.COLOR_RGB2YUV)
    # Find blobs.
    blob_centers = {}
    blob_visualizations = {}
    for name, color_range in self._color_ranges.items():
      blob = self._find_blob(yuv_image, color_range)
      blob_centers[name] = blob.center * (1. / self._scale) if blob else None
      blob_visualizations[name] = (
          self._draw_blobs(image, blob) if self._visualize else None)
    if self._toolkit:
      self._update_gui_toolkit(yuv_image, image)
    return blob_centers, blob_visualizations

  def _preprocess(self, image: np.ndarray) -> np.ndarray:
    """Preprocesses an image for color-based blob detection."""
    # Resize the image to make all other operations faster.
    size = np.round(image.shape[:2] * self._scale).astype(np.int32)
    resized = cv2.resize(image, (size[1], size[0]))
    if self._mask is None:
      self._setup_mask(resized)
    # Denoise the image.
    denoised = cv2.fastNlMeansDenoisingColored(
        src=resized, h=7, hColor=7, templateWindowSize=3, searchWindowSize=5)
    return cv2.multiply(denoised, self._mask)

  def _setup_mask(self, image: np.ndarray) -> None:
    """Initialises an image mask to explude pixels from blob detection."""
    self._mask = np.ones(image.shape, image.dtype)
    for mask_points in self._mask_points:
      cv2.fillPoly(self._mask, np.int32([mask_points * self._scale]), 0)

  def _find_blob(self, yuv_image: np.ndarray,
                 color_range: types.ColorRange) -> Optional[types.Blob]:
    """Find the largest blob matching the YUV color range.

    Args:
      yuv_image: An image in YUV color space.
      color_range: The YUV color range used for segmentation.

    Returns:
      If found, the (u, v) coordinate of the barycenter and the contour of the
      segmented blob. Otherwise returns `None`.
    """
    # Threshold the image in YUV color space.
    lower = color_range.lower
    upper = color_range.upper
    mask = cv2.inRange(yuv_image.copy(), lower, upper)
    # Find contours.
    contours, _ = cv2.findContours(
        image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
      return None
    # Find the largest contour.
    max_area_contour = max(contours, key=cv2.contourArea)
    # If the blob's area is too small, ignore it.
    correction_factor = np.square(1. / self._scale)
    normalized_area = cv2.contourArea(max_area_contour) * correction_factor
    if normalized_area < self._min_area:
      return None
    # Compute the centroid.
    moments = cv2.moments(max_area_contour)
    if moments["m00"] == 0:
      return None
    cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    return types.Blob(center=np.array([cx, cy]), contour=max_area_contour)

  def _draw_blobs(self, image: np.ndarray, blob: types.Blob) -> np.ndarray:
    """Draws the controuer of the detected blobs."""
    frame = image.copy()
    if blob:
      # Draw center.
      cv2.drawMarker(
          img=frame,
          position=(int(blob.center[0]), int(blob.center[1])),
          color=(255, 0, 0),
          markerType=cv2.MARKER_CROSS,
          markerSize=7,
          thickness=1,
          line_type=cv2.LINE_AA)
      # Draw contours.
      cv2.drawContours(
          image=frame,
          contours=[blob.contour],
          contourIdx=0,
          color=(0, 0, 255),
          thickness=1)
    return frame

  def _callback_change_color(self, color_index: int) -> None:
    """Callback for YUV GUI toolkit trackbar.

    Reads current trackbar value and selects the associated color.
    The association between index and color is implementation dependent, i.e.
    in the insertion order into a dictionary.

    Args:
      color_index: The current value of the trackbar. Passed automatically.
    """
    colors = list(self._color_ranges.keys())
    selected_color = colors[color_index]
    min_upper = self._color_ranges[selected_color]
    lower = min_upper.lower
    upper = min_upper.upper

    self._u_range_trackbar.set_trackbar_pos(lower[1], upper[1])
    self._v_range_trackbar.set_trackbar_pos(lower[2], upper[2])

    cv2.setWindowTitle(self._window_name,
                       self._window_name + " - Color: " + selected_color)

  def _update_gui_toolkit(self, image_yuv: np.ndarray,
                          image_rgb: np.ndarray) -> None:
    """Updates the YUV GUI toolkit.

    Creates and shows the UV representation of the current image.

    Args:
      image_yuv: The current image in YUV color space.
      image_rgb: The current image in RGB color space.
    """
    subsample = cv2.getTrackbarPos("Subsampling", self._window_name)
    img_u = image_yuv[0::subsample, 0::subsample, 1]
    img_v = 1.0 - image_yuv[0::subsample, 0::subsample, 2]
    pixel_color = image_rgb[0::subsample, 0::subsample, :]

    pixel_color = pixel_color.reshape(np.prod(img_u.shape[0:2]), -1)
    img_u = img_u.ravel()
    img_v = img_v.ravel()

    fig_size = 300
    fig = np.full(shape=(fig_size, fig_size, 3), fill_value=255, dtype=np.uint8)
    cv2.arrowedLine(
        img=fig,
        pt1=(0, fig_size),
        pt2=(fig_size, fig_size),
        color=(0, 0, 0),
        thickness=2,
        tipLength=0.03)
    cv2.arrowedLine(
        img=fig,
        pt1=(0, fig_size),
        pt2=(0, 0),
        color=(0, 0, 0),
        thickness=2,
        tipLength=0.03)
    cv2.putText(
        img=fig,
        text="U",
        org=(int(0.94 * fig_size), int(0.97 * fig_size)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        thickness=2)
    cv2.putText(
        img=fig,
        text="V",
        org=(int(0.03 * fig_size), int(0.06 * fig_size)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        thickness=2)

    for i in range(img_u.size):
      color = tuple(int(p) for p in pixel_color[i, ::-1])

      position = (int(img_u[i] * fig_size), int(img_v[i] * fig_size))
      cv2.drawMarker(
          img=fig,
          position=position,
          color=color,
          markerType=cv2.MARKER_SQUARE,
          markerSize=3,
          thickness=2)

    u_min, u_max = self._u_range_trackbar.get_trackbar_pos()
    u_min = int(u_min * fig_size)
    u_max = int(u_max * fig_size)
    v_min, v_max = self._v_range_trackbar.get_trackbar_pos()
    v_min = int((1.0 - v_min) * fig_size)
    v_max = int((1.0 - v_max) * fig_size)

    cv2.line(
        img=fig,
        pt1=(u_min, v_max),
        pt2=(u_min, v_min),
        color=(0, 0, 0),
        thickness=2)
    cv2.line(
        img=fig,
        pt1=(u_max, v_max),
        pt2=(u_max, v_min),
        color=(0, 0, 0),
        thickness=2)
    cv2.line(
        img=fig,
        pt1=(u_min, v_min),
        pt2=(u_max, v_min),
        color=(0, 0, 0),
        thickness=2)
    cv2.line(
        img=fig,
        pt1=(u_min, v_max),
        pt2=(u_max, v_max),
        color=(0, 0, 0),
        thickness=2)

    cv2.imshow(self._window_name, fig)
    cv2.waitKey(1)


class CreateRangeTrackbar:
  """Class to create and control, on an OpenCV GUI, two trackbars representing a range of values."""

  def __init__(self,
               window_name: str,
               trackbar_name_lower: str,
               trackbar_name_upper: str,
               color_ranges: Mapping[str, types.ColorRange],
               color_code: str,
               trackbar_scale: int = 1000):
    """Initializes the class.

    Args:
      window_name: Name of the window that will be used as a parent of the
        created trackbar.
      trackbar_name_lower: The name of the trackbar implementing the lower bound
        of the range.
      trackbar_name_upper: The name of the trackbar implementing the upper bound
        of the range.
      color_ranges: A mapping between a given blob name and the range of YUV
        color used to segment it from an image.
      color_code: The color code to change in `color_ranges`. Shall be "U" or
        "V".
      trackbar_scale: The trackbar scale to recover the real value from the
        current trackbar position.
    """
    self._window_name = window_name
    self._trackbar_name_lower = trackbar_name_lower
    self._trackbar_name_upper = trackbar_name_upper
    self._color_ranges = color_ranges
    self._color_code = color_code
    self._trackbar_scale = trackbar_scale
    self._trackbar_reset = False

    # pylint: disable=g-long-lambda
    cv2.createTrackbar(
        self._trackbar_name_lower, self._window_name, 0,
        self._trackbar_scale, lambda x: self._callback_update_threshold(
            "lower", "lower", self._color_code, x))
    cv2.createTrackbar(
        self._trackbar_name_upper, self._window_name, 0,
        self._trackbar_scale, lambda x: self._callback_update_threshold(
            "upper", "upper", self._color_code, x))
    # pylint: enable=g-long-lambda

  def set_trackbar_pos(self, lower_value: float, upper_value: float) -> None:
    """Sets the trackbars to specific values."""
    if lower_value > upper_value:
      logging.error(
          "Wrong values for setting range trackbars. Lower value "
          "must be less than upper value. Provided lower: %d. "
          "Provided upper: %d.", lower_value, upper_value)
      return

    # To change the trackbar values avoiding the consistency check enforced by
    # the callback to implement a range of values with two sliders, we set the
    # variable self._trackbar_reset to `True` and then bring it back to
    # `False`.

    self._trackbar_reset = True
    cv2.setTrackbarPos(self._trackbar_name_lower, self._window_name,
                       int(lower_value * self._trackbar_scale))
    cv2.setTrackbarPos(self._trackbar_name_upper, self._window_name,
                       int(upper_value * self._trackbar_scale))
    self._trackbar_reset = False

  def get_trackbar_pos(self, normalized: bool = True) -> Tuple[float, float]:
    """Gets the trackbars lower and upper values."""
    lower = cv2.getTrackbarPos(self._trackbar_name_lower, self._window_name)
    upper = cv2.getTrackbarPos(self._trackbar_name_upper, self._window_name)
    if normalized:
      return lower / self._trackbar_scale, upper / self._trackbar_scale
    else:
      return lower, upper

  def _callback_update_threshold(self, lower_or_upper: str, attribute: str,
                                 color_code: str, value: int) -> None:
    """Callback for YUV GUI toolkit trackbar.

    Reads current trackbar value and updates the associated U or V threshold.
    This callback assumes that two trackbars, `trackbar_name_lower` and
    `trackbar_name_upper`, form a range of values. As a consequence, when one
    of the two trackbar is moved, there is a consistency check that the range
    is valid (i.e. lower value less than max value and vice versa).

    Typical usage example:
      To pass it to an OpenCV/Qt trackbar, use this function in a lambda
      as follows:
      cv2.createTrackbar("Trackbar lower", ..., lambda x:
      class_variable._callback_update_threshold("lower", "lower", "U", x))

    Args:
      lower_or_upper: The behaviour of this callback for the range. Shall be
        `lower` or `upper`.
      attribute: The name of the threshold in `self._color_ranges` for the
        current selected color.
      color_code: The color code to change. Shall be "U" or "V".
      value: The current value of the trackbar.
    """
    if not self._trackbar_reset:
      if lower_or_upper == "lower":
        limiting_value = cv2.getTrackbarPos(self._trackbar_name_upper,
                                            self._window_name)
        if value > limiting_value:
          cv2.setTrackbarPos(self._trackbar_name_lower, self._window_name,
                             limiting_value)
          return
      elif lower_or_upper == "upper":
        limiting_value = cv2.getTrackbarPos(self._trackbar_name_lower,
                                            self._window_name)
        if value < limiting_value:
          cv2.setTrackbarPos(self._trackbar_name_upper, self._window_name,
                             limiting_value)
          return

    selected_color_index = cv2.getTrackbarPos("Color selector",
                                              self._window_name)
    colors = list(self._color_ranges.keys())
    selected_color = colors[selected_color_index]
    updated_value = value / self._trackbar_scale

    color_threshold = getattr(self._color_ranges[selected_color], attribute)
    if color_code == "U":
      color_threshold[1] = updated_value
    elif color_code == "V":
      color_threshold[2] = updated_value
    else:
      logging.error(
          "Wrong trackbar name. No U/V color code correspondence."
          "Provided: `%s`.", color_code)
      return
