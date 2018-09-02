"""
The code in this file was copied from https://bitbucket.org/amitibo/pyfisheye

Copyright (c) 2015 Amit Aides.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import json
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
from joblib import Parallel, delayed
import logging
from numpy import finfo
import numpy as np
import os
import pickle

eps = finfo(np.float).eps



def extract_corners(img, img_index, nx, ny, subpix_criteria, verbose):
    """Extract chessboard corners."""

    if type(img) == str:
        fname = img
        if verbose:
            logging.info("Processing img: {}...".format(os.path.split(fname)[1]))

        #
        # Load the image.
        #
        img = cv2.imread(fname)
    else:
        if verbose:
            logging.info("Processing img: {}...".format(img_index))

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    #
    # Find the chess board corners
    #
    ret, cb_2D_pts = cv2.findChessboardCorners(
        gray,
        (nx, ny),
        cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        #
        # Refine the corners.
        #
        cb_2D_pts = cv2.cornerSubPix(
            gray,
            cb_2D_pts,
            (3, 3),
            (-1, -1),
            subpix_criteria
        )

    return ret, cb_2D_pts


class FishEyeCamera(object):
    """Fisheye Camera Class

    Wrapper around the opencv fisheye calibration code.

    Args:
        nx, ny (int): Number of inner corners of the chessboard pattern, in x and y axes.
        verbose (bool): Verbose flag.
    """

    def __init__(self, K=np.zeros((3, 3)), D=np.zeros((4, 1)), R=np.eye(3), M=np.zeros((3,3), dtype=np.float),
                 img_shape=None, verbose=False):

        self._verbose = verbose

        if type(K) is not np.array:
            K = np.array(K)

        if type(D) is not np.array:
            D = np.array(D)

        if type(R) is not np.array:
            R = np.array(R)

        if type(M) is not np.array:
            M = np.array(M)

        self._K = K
        self._D = D
        self._R = R
        self._M = M
        self._img_shape = img_shape



    def calibrate_undistort(
        self,
        nx,
        ny,
        img_paths=None,
        imgs=None,
        update_model=True,
        max_iter=30,
        eps=1e-6,
        show_imgs=False,
        return_mask=False,
        calibration_flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW,
        n_jobs=-1,
        backend='threading'
        ):
        """Calibration

        Calibrate a fisheye model using images of chessboard pattern.

        Args:
            nx, ny (int): Number of inner corners of the chessboard pattern, in x and y axes.
            img_paths (list of paths): Paths to images of chessboard pattern.
            update_model (optional[bool]): Whether to update the stored clibration. Set to
                False when you are interested in calculating the position of
                chess boards.
            max_iter (optional[int]): Maximal iteration number. Defaults to 30.
            eps (optional[int]): error threshold. Defualts to 1e-6.
            show_imgs (optional[bool]): Show calibtration images.
            calibration_flags (optional[int]): opencv flags to use in the opencv.fisheye.calibrate command.
        """

        assert not ((img_paths is None) and (imgs is None)), 'Either specify imgs or img_paths'

        self.chessboard_model = np.zeros((1, nx*ny, 3), np.float32)
        self.chessboard_model[0, :, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        #
        # Arrays to store the chessboard image points from all the images.
        #
        chess_2Dpts_list = []

        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        if show_imgs:
            cv2.namedWindow('checkboard img', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('fail img', cv2.WINDOW_AUTOSIZE)

        if img_paths is not None:
            imgs = img_paths

        with Parallel(n_jobs=n_jobs, backend=backend) as parallel:
            rets = parallel(
                delayed(extract_corners)(
                    img, img_index, nx, ny, subpix_criteria, self._verbose
                    ) for img_index, img in enumerate(imgs)
            )

        mask = []
        for img_index, (img, (ret, cb_2d_pts)) in enumerate(zip(imgs, rets)):

            if type(img) == str:
                fname = img
                if self._verbose:
                    logging.info("Processing img: {}...".format(os.path.split(fname)[1]))

                #
                # Load the image.
                #
                img = cv2.imread(fname)
            else:
                if self._verbose:
                    logging.info("Processing img: {}...".format(img_index))

            if self._img_shape == None:
                self._img_shape = img.shape[:2]
            else:
                assert self._img_shape == img.shape[:2], \
                       "All images must share the same size."
            if ret:
                mask.append(True)

                #
                # Was able to find the chessboard in the image, append the 3D points
                # and image points (after refining them).
                #
                if self._verbose:
                    logging.info('OK')

                #
                # The 2D points are reshaped to (1, N, 2). This is a hack to handle the bug
                # in the opecv python wrapper.
                #
                chess_2Dpts_list.append(cb_2d_pts.reshape(1, -1, 2))

                if show_imgs:
                    #
                    # Draw and display the corners
                    #
                    img = cv2.drawChessboardCorners(
                        img.copy(), (nx, ny),
                        cb_2d_pts,
                        ret
                    )
                    cv2.imshow('checkboard img', img)
                    cv2.waitKey(500)
            else:
                mask.append(False)

                if self._verbose:
                    logging.info('FAIL!')

                if show_imgs:
                    #
                    # Show failed img
                    #
                    cv2.imshow('fail img', img)
                    cv2.waitKey(500)

        if show_imgs:
            #
            # Clean up.
            #
            cv2.destroyAllWindows()

        N_OK = len(chess_2Dpts_list)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        #
        # Update the intrinsic model
        #
        if update_model:
            K = self._K
            D = self._D
        else:
            K = self._K.copy()
            D = self._D.copy()

        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                [self.chessboard_model]*N_OK,
                chess_2Dpts_list,
                (img.shape[1], img.shape[0]),
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
            )

        if return_mask:
            return rms, K, D, rvecs, tvecs, mask
        else:
            return rms, K, D, rvecs, tvecs

    @staticmethod
    def scale_K(K, calib_img_shape, source_img_shape):
        """
        Get K for a different size image.
        
        Inputs:
         - calib_img_shape: tuple in the format (x_pixels, y_pixels)
         - source_img_shape: tuple in the format (x_pixels, y_pixels)
        
        """
        x_scale = source_img_shape[0] / calib_img_shape[0]
        y_scale = source_img_shape[1] / calib_img_shape[1]

        assert x_scale == y_scale

        # scale only the top two rows of K.
        scales = np.array([x_scale, x_scale, 1])
        K_scaled = (K.T * scales).T
        return K_scaled

    def update_img_size(self, new_img):
        """

        :param new_img:
        :return:
        """
        self._K = self.scale_K(self._K, self._img_shape, new_img.shape)
        self._img_shape = new_img.shape
        return


    def undistort(self, distorted_img, undistorted_size=None, R=None, K=None):
        """Undistort an image using the fisheye model"""

        if R is None:
            R = self._R

        if K is None:
            K = self._K

        if undistorted_size is None:
            undistorted_size = (distorted_img.shape[1], distorted_img.shape[0])


        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self._K,
            self._D,
            R,
            K,
            undistorted_size,
            cv2.CV_16SC2
        )

        undistorted_img = cv2.remap(
            distorted_img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        return undistorted_img


    def projectPoints(self, object_points=None, skew=0, rvec=None, tvec=None):
        """Projects points using fisheye model.
        """

        if object_points is None:
            #
            # The default is to project the checkerboard.
            #
            object_points = self.chessboard_model

        if object_points.ndim == 2:
            object_points = np.expand_dims(object_points, 0)

        if rvec is None:
            rvec = np.zeros(3).reshape(1, 1, 3)
        else:
            rvec = np.array(rvec).reshape(1, 1, 3)

        if tvec is None:
            tvec = np.zeros(3).reshape(1, 1, 3)
        else:
            tvec = np.array(tvec).reshape(1, 1, 3)

        image_points, jacobian = cv2.fisheye.projectPoints(
            object_points,
            rvec,
            tvec,
            self._K,
            self._D,
            alpha=skew
        )

        return np.squeeze(image_points)

    def undistortPoints(self, distorted, R=None, K=None):
        """Undistorts 2D points using fisheye model.
        """

        if distorted.ndim == 2:
            distorted = np.expand_dims(distorted, 0)

        if R is None:
            R = self._R

        if K is None:
            K = self._K

        undistorted = cv2.fisheye.undistortPoints(
            distorted.astype(np.float32),
            self._K,
            self._D,
            R=R,
            P=K
        )

        return np.squeeze(undistorted)

    def undistortDirections(self, distorted):
        """Undistorts 2D points using fisheye model.

        Args:
            distorted (array): nx2 array of distorted image coords (x, y).

        Retruns:
            Phi, Theta (array): Phi and Theta undistorted directions.
        """

        assert distorted.ndim == 2 and distorted.shape[1] == 2, "distorted should be nx2 points array."

        #
        # Calculate
        #
        f = np.array((self._K[0,0], self._K[1,1])).reshape(1, 2)
        c = np.array((self._K[0,2], self._K[1,2])).reshape(1, 2)
        k = self._D.ravel().astype(np.float64)

        #
        # Image points
        #
        pi = distorted.astype(np.float)

        #
        # World points (distorted)
        #
        pw = (pi-c)/f

        #
        # Compensate iteratively for the distortion.
        #
        theta_d = np.linalg.norm(pw, ord=2, axis=1)
        theta = theta_d
        for j in range(10):
            theta2 = theta**2
            theta4 = theta2**2
            theta6 = theta4*theta2
            theta8 = theta6*theta2
            theta = theta_d / (1 + k[0]*theta2 + k[1]*theta4 + k[2]*theta6 + k[3]*theta8)

        theta_d_ = theta * (1 + k[0]*theta**2 + k[1]*theta**4 + k[2]*theta**6 + k[3]*theta**8)

        #
        # Mask stable theta values.
        #
        ratio = np.abs(theta_d_-theta_d)/(theta_d+eps)
        mask = (ratio < 1e-2)

        #
        # Scale is equal to \prod{\r}{\theta_d} (http://docs.opencv.org/trunk/db/d58/group__calib3d__fisheye.html)
        #
        scale = np.tan(theta) / (theta_d + eps)

        #
        # Undistort points
        #
        pu = pw * scale.reshape(-1, 1)
        phi = np.arctan2(pu[:, 0], pu[:, 1])

        return phi, theta, mask

    def calibrate_birdseye(self, src_points, dst_points):

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        self._M = M
        return M

    def birdseye(self, img):
        img_size = (self._img_shape[1], self._img_shape[0]) #convert to opencv ordering
        return cv2.warpPerspective(img, self._M, dsize=img_size, flags=cv2.INTER_LINEAR)




    def save(self, filename):
        """Save the fisheye model."""

        with open(filename, 'w') as f:
            json.dump(self.dump_params(), f)


    @property
    def img_shape(self):
        """Shape of image used for calibration."""

        return self._img_shape

    def dump_params(self):
        """
        Return dictionary of parameters required to load Fisheye class.
        """
        params = {
            "K": self._K.tolist(),
            "D": self._D.tolist(),
            "R": self._R.tolist(),
            "M": self._M.tolist(),
            "img_shape": self._img_shape,
        }
        return params


    @classmethod
    def load(cls, params_dict=None, params_path=None):

        if params_dict:
            pass
        elif params_path:
            with open(params_path, 'r') as f:
                params_dict = json.load(f)

        obj = FishEyeCamera(**params_dict)
        return obj
