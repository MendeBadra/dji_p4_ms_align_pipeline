#!/usr/bin/env python
# coding: utf-8

## python raw2reflectance.py input_dir output_dir

# Radiometric Calibration with Sun Irradiance Sensor (= DLS Sensor)
# Raw Image (.TIF) --> Reflectance Image (.TIF, Scale: 0-2**16)
# Reflectance Image / 2**16 --> 0-1 value

import os
import glob
import numpy as np
import tifffile
import micasense.capture as capture
import micasense.dls as dls
import micasense.metadata as metadata
import time
import sys
import exiftool # This library might need to be installed separately, often via pip or by having ExifTool installed on your system and its path configured.
import json

def raw_to_reflectance(input_dir, output_dir):
    """
    Performs radiometric calibration on raw MicaSense RedEdge images (.TIF)
    using DLS sensor data and converts them to reflectance images.
    Metadata is copied from the raw images to the generated reflectance images.

    Args:
        input_dir (str): The path to the directory containing raw .TIF images.
        output_dir (str): The path to the directory where reflectance .TIF images will be saved.
    """
    start = time.time()
    initial_path = os.getcwd()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    raw_image_list = sorted(glob.glob(os.path.join(input_dir, '*.TIF')))

    if not raw_image_list:
        print(f"No .TIF images found in the input directory: {input_dir}")
        return

    print("---")
    print("Start Radiometric Calibration")
    print("---")

    for i, raw_image_path in enumerate(raw_image_list):
        print(f"Processing {os.path.basename(raw_image_path)} [{i+1}/{len(raw_image_list)}] {((i+1)/len(raw_image_list))*100:.2f}% done.")
        print(raw_image_path)

        cap = capture.Capture.from_filelist([raw_image_path])

        # Define DLS sensor orientation vector relative to dls pose frame
        dls_orientation_vector = np.array([0, 0, -1])

        # Compute sun orientation and sun-sensor angles
        (
            sun_vector_ned,    # Solar vector in North-East-Down coordinates
            sensor_vector_ned, # DLS vector in North-East-Down coordinates
            sun_sensor_angle,  # Angle between DLS vector and sun vector
            solar_elevation,   # Elevation of the sun above the horizon
            solar_azimuth,     # Azimuth (heading) of the sun
        ) = dls.compute_sun_angle(cap.location(),
                                  cap.dls_pose(),
                                  cap.utc_time(),
                                  dls_orientation_vector)

        # Get Spectral Irradiance from metadata
        meta = metadata.Metadata(raw_image_path, exiftoolPath=None)
        spectral_irradiance = meta.get_item('XMP:Irradiance')
        print(f"Spectral Irradiance: {spectral_irradiance}")

        # Correct the raw sun sensor irradiance value (DLS) and compute the irradiance on level ground
        fresnel_correction = dls.fresnel(sun_sensor_angle)
        dir_dif_ratio = 6.0  # Default value from MicaSense
        percent_diffuse = 1.0 / dir_dif_ratio
        sensor_irradiance = spectral_irradiance / fresnel_correction
        untilted_direct_irr = sensor_irradiance / (percent_diffuse + np.cos(sun_sensor_angle))

        # Compute irradiance on the ground (= DLS Irradiance) using the solar altitude angle
        dls_irr = untilted_direct_irr * (percent_diffuse + np.sin(solar_elevation))
        dls_irradiances = [dls_irr]

        # Produce Reflectance Images
        reflectance_image = cap.compute_reflectance(dls_irradiances)
        reflectance_image = np.array(reflectance_image)[0, :, :].astype(np.float32)

        # Scale to uint16
        refl_uint16 = np.round(reflectance_image * (2**16 - 1)).astype(np.uint16)

        # Save Reflectance Images
        filename = os.path.splitext(os.path.basename(raw_image_path))[0]
        output_filepath = os.path.join(output_dir, f'{filename}.TIF')
        tifffile.imwrite(output_filepath, refl_uint16)

    print(f"---")
    print(f"{len(raw_image_list)} reflectance images are created in {output_dir}")
    print(f"---")

    print(f"---")
    print("Start Copying Metadata")
    print(f"---")

    # Re-list images to ensure correct pairing after potential sorting differences
    raw_image_list = sorted(glob.glob(os.path.join(input_dir, '*.TIF')))
    ref_image_list = sorted(glob.glob(os.path.join(output_dir, '*.TIF')))

    if len(raw_image_list) != len(ref_image_list):
        print("Warning: Number of raw images does not match number of reflectance images. Metadata copying might be inaccurate.")

    for i in range(len(raw_image_list)):
        raw_image = raw_image_list[i]
        ref_image = ref_image_list[i] # Assuming corresponding names after sorting

        print(f"Processing metadata for {os.path.basename(raw_image)} [{i+1}/{len(raw_image_list)}] {((i+1)/len(raw_image_list))*100:.2f}% done.")

        if os.path.basename(raw_image) != os.path.basename(ref_image):
            print(f"Warning: Filename mismatch detected. Raw: {os.path.basename(raw_image)}, Reflectance: {os.path.basename(ref_image)}. Metadata might not be copied to the correct file.")

        # Copy all metadata except XMP from raw to reflectance image
        os.system(f'exiftool -TagsFromFile "{raw_image}" -all:all -xmp -overwrite_original "{ref_image}"')
        # Set Blacklevel to 0 (important for reflectance images)
        os.system(f'exiftool -IFD0:Blacklevel=0 -overwrite_original "{ref_image}"')

    print(f"---")
    print(f"Metadata copying complete.")
    print(f"It took {time.time()-start:.2f} seconds to complete the process.")
    print(f"---")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python your_script_name.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    raw_to_reflectance(input_directory, output_directory)