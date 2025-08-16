import requests
import os

def download_images(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    image_urls = []
    print(f"Enter image URLs to save in directory '{save_dir}' (` to start a new entry):")

    while True:
        url = input("Enter image URL: ").strip()
        if url == '`':
            print("Starting a new entry.")
            return  # Exit the current entry and prompt for new directory name
        # if url == '0':
            # print("Input stopped.")
            # break
        else:
            image_urls.append(url)

        # Downloading images
        for i, url in enumerate(image_urls):
            try:
                response = requests.get(url)
                response.raise_for_status()
                ext = url.split('.')[-1].split('?')[0]
                filename = f'image_{i + 1}.{ext}'
                with open(os.path.join(save_dir, filename), 'wb') as f:
                    f.write(response.content)
                print(f'Downloaded {filename}')
            except Exception as e:
                print(f'Failed to download {url}: {e}')

# Main loop for app
while True:
    save_dir = input("\nDIR name:: ").strip()
    # if save_di/r == '`':
        # print("Exiting application.")
        # break
    download_images(save_dir)

# import requests
# import os

# # Directory to save images
# save_dir = 'Nimoy'
# os.makedirs(save_dir, exist_ok=True)

# image_urls = []
# print("Enter image URLs one by one (enter 0 to stop):")

# while True:
#     url = input("Enter image URL: ").strip()
#     if url == '`':
#         print("Input stopped.")
#         break
#     if url:
#         image_urls.append(url)

# # Downloading images
# for i, url in enumerate(image_urls):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         ext = url.split('.')[-1].split('?')[0]
#         filename = f'image_{i + 1}.{ext}'
#         with open(os.path.join(save_dir, filename), 'wb') as f:a
#             f.write(response.content)
#         print(f'Downloaded {filename}')
#     except Exception as e:
#         print(f'Failed to download {url}: {e}')
