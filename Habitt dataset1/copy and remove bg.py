import os
import random
import cv2
import numpy as np

# Get the current working directory
current_dir = os.getcwd()

# Function to remove white background
def remove_white_background(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert to BGRA (adds alpha channel)
    if image.shape[2] != 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Create mask for near-white pixels
    white_mask = np.all(image[:, :, :3] >= [240, 240, 240], axis=2)

    # Set alpha to 0 for white areas
    image[white_mask, 3] = 0

    return image

# # Loop through each item in the current directory
# for folder_name in os.listdir(current_dir):
#     subfolder_path = os.path.join(current_dir, folder_name)

#     # Proceed only if it's a directory
#     if os.path.isdir(subfolder_path):
#         # Get list of .jpg files
#         jpg_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.jpg')]

#         if jpg_files:
#             # Pick one .jpg at random
#             selected_file = random.choice(jpg_files)
#             source_path = os.path.join(subfolder_path, selected_file)
#             output_path = os.path.join(current_dir, f"{folder_name}.png")  # Save as subfolder name

#             # Remove background and save
#             result_img = remove_white_background(source_path)
#             cv2.imwrite(output_path, result_img)

#             print(f"Saved: {output_path}")
#         else:
#             print(f"No .jpg files found in {subfolder_path}")


# Remove background and save
result_img = remove_white_background(r"C:\OneDrive - Higher Education Commission\other data\System Folders\Downloads\app.png")
cv2.imwrite(r"C:\OneDrive - Higher Education Commission\other data\System Folders\Downloads\app 1.png", result_img)

# print(f"Saved: {output_path}")
# else:
    # print(f"No .jpg files found in {subfolder_path}")






# import os

# # Get the current working directory
# current_dir = os.getcwd()

# # List all .png files in the current directory
# png_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.png')]

# # Print the file names
# for filename in png_files:
#     print(filename)




# import os
# import shutil

# # Get the current working directory
# current_dir = os.getcwd()

# # List all .png files in the current directory
# png_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.png')]

# for png_file in png_files:
#     # Get base name without extension
#     folder_name = os.path.splitext(png_file)[0]
#     folder_path = os.path.join(current_dir, folder_name)

#     # Create the subfolder if it doesn't exist
#     os.makedirs(folder_path, exist_ok=True)

#     # Define source and destination paths
#     src_path = os.path.join(current_dir, png_file)
#     dst_path = os.path.join(folder_path, png_file)

#     # Copy the file
#     shutil.copy2(src_path, dst_path)

#     print(f"Copied {png_file} to {folder_path}/")
