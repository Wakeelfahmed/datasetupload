%%writefile FurnitureAPI.py
import torch
from flask import Flask, send_file, request, jsonify
import traceback  # Import the traceback module for error logging
import base64  # Import the base64 module for encoding and decoding binary data
import os
import json  # Import the json module to work with JSON data
import sqlite3
import csv
import io
import subprocess
import open3d as o3d

#Trellis
import os
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.

import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils  # Added postprocessing_utils

# Load the pipeline
device_pipeline = torch.device("cuda:0")
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device_pipeline)
# pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
# pipeline.cuda(torch.device("cuda:0"))
# pipeline.cuda()
# ----

from trellis.pipelines import TrellisTextTo3DPipeline
# # Load the text-to-3D texture pipeline
# texture_pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
# texture_pipeline.cuda(torch.device("cuda:1"))
# texture_pipeline.cuda()

device_texture = torch.device("cuda:1")
texture_pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
texture_pipeline.to(device_texture)



app = Flask(__name__)

database_name = 'Server.s3db'

# Database configuration
@app.route('/insert_data', methods=['POST'])
def initialize_database():
    if not os.path.exists(database_name):
        print(f"Creating a new database: {database_name}")
        # Initialize the database schema and perform any setup if needed
        conn = sqlite3.connect(database_name)
        cursor = conn.cursor()
        # Create tables and perform any necessary setup here
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS item (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT,
                price REAL,
                url TEXT,
                desc TEXT,
                categories TEXT,
                brands TEXT,
                designer TEXT,
                spec TEXT,
                noClick INTEGER,
                downloaded INTEGER
            )
        """)

        conn.commit()
        conn.close()
    else:
        print(f"Using existing database: {database_name}")


    CSVdata = request.form['CSV']
    print("initialize_database CSV " + CSVdata)
    csv_file = io.StringIO(CSVdata)
    # Create a CSV reader from the file-like object
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        print("initialize_database Data from CSV string:", row)
        conn = sqlite3.connect(database_name)
        cursor = conn.cursor()

        # Extract data from the row
        name = row[0]
        price = row[1]
        print("3:" + row[3])
        description = row[2]
        brand = row[3]
        print("2:" + row[2])

        url = row[4]
        designer = row[5]
        spec = row[6]
#         categories = row[7:]  # The remaining elements are categories
        categories = [cat.strip() for cat in row[7:] if cat.strip() != '']

        #Insert the data into the database
        insert_query = """
         INSERT INTO item (Name, price, url, desc, categories, brands, designer, spec, noClick, downloaded)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
         """
    print("initialize_database The insert query is " + str(insert_query))
    # Create a tuple containing the data to be inserted
    data_tuple = (name, price, url, description, ', '.join(categories), brand, designer, spec, 0, 0)
    print("initialize_database The insert query is " + str(data_tuple))
    cursor.execute(insert_query, data_tuple)
    conn.commit()
    conn.close()
    return "Data inserted successfully"

    
@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        Itemname = request.form['Itemname']
        print("Itemname: " + Itemname)
        uploaded_files = request.files.getlist('images')
        file_names = [file.filename for file in uploaded_files]  # Extract the filenames

        print("Uploaded files: " + ', '.join(file_names))  # Join the filenames into a single string
        
        upload_path = os.path.join('Received Images', Itemname, 'Images')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
#         if not os.path.exists('Received Images\\'+ Itemname + "\\Images"):
#             os.makedirs('Received Images\\'+ Itemname + "\\Images")

        for image_file in uploaded_files:
            if image_file.filename != '':
                # Construct a full file path to save the image
                file_path = os.path.join('Received Images', Itemname , "Images", image_file.filename)
                # Save the image to the specified path
                image_file.save(file_path)
                print(f"Saved image to: {file_path}")
                # print("Saved image to: "+file_path)
            else:
                print("image_file.filename != ''" + file_path)


        return 'Images uploaded successfully'

    except Exception as e:
        print(f"Error during image upload: {str(e)}")
        return 'Image upload failed'
 

import os
import cv2
import numpy as np
from flask import Flask, request
from PIL import Image

def remove_background_cv2(image_path):
    # Read image with alpha channel support
    image = cv2.imread(image_path)
    
    # Convert to grayscale and threshold to create mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Convert mask to 3 channels
    mask_3ch = cv2.merge([mask, mask, mask])

    # Apply mask to image
    result = cv2.bitwise_and(image, mask_3ch)

    # Add alpha channel to result
    b, g, r = cv2.split(result)
    alpha = mask
    rgba = cv2.merge([b, g, r, alpha])

    return rgba


@app.route('/SavePreviewImage', methods=['POST'])
def SavePreviewImage():
    try:
        Itemname = request.form['Itemname']
        uploaded_file = request.files['image']  # Use 'image' as the file field name

        if uploaded_file.filename != '':
            item_path = os.path.join('Received Images', Itemname)
            os.makedirs(item_path, exist_ok=True)

            # Ensure the file is saved temporarily
            original_path = os.path.join(item_path, uploaded_file.filename)
            uploaded_file.save(original_path)

            # Remove background using cv2
            result_image = remove_background_cv2(original_path)

            # Change extension to .png (even if uploaded as .jpg etc.)
            name_wo_ext = os.path.splitext(uploaded_file.filename)[0]
            output_path = os.path.join(item_path, name_wo_ext + '.png')

            # Save image with alpha channel
            cv2.imwrite(output_path, result_image)

            print(f"Saved image to: {output_path}")
            return 'Image uploaded and processed successfully'

    except Exception as e:
        print(f"Error during image upload: {str(e)}")
        return 'Image upload failed'


@app.route('/add_texture', methods=['POST'])
def add_texture():
    try:
        itemname = request.form.get('Itemname')
        prompt = request.form.get('prompt')

        if not itemname or not prompt:
            return jsonify({"error": "Missing Itemname or prompt"}), 400

        print(f"Received texture generation request for {itemname} with prompt: {prompt}")

        # Locate base mesh (PLY file)
        base_mesh_path = os.path.join("TRELLIS Out Models", itemname, f"{itemname}.ply")
        if not os.path.exists(base_mesh_path):
            return jsonify({"error": "Base mesh not found"}), 404

        # Load the base mesh
        base_mesh = o3d.io.read_triangle_mesh(base_mesh_path)
        base_mesh = base_mesh.to(device_texture)


        # Run texture pipeline
        outputs = texture_pipeline.run_variant(
            base_mesh,
            prompt,
            seed=1,
        )

        # Save outputs
        output_folder = os.path.join("TRELLIS Out Models", itemname)
        os.makedirs(output_folder, exist_ok=True)

        # Save updated mesh with texture
        textured_mesh_path = os.path.join(output_folder, f"{itemname}_textured.ply")
        o3d.io.write_triangle_mesh(textured_mesh_path, outputs['mesh'][0])

        # Save render video
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([f_gs, f_mesh], axis=1) for f_gs, f_mesh in zip(video_gs, video_mesh)]

        video_path = os.path.join(output_folder, "textured_output.mp4")
        imageio.mimsave(video_path, video, fps=30)

        print(f"Textured mesh and video saved for {itemname}")

        return jsonify({"message": "Texture generation completed successfully"}), 200

    except Exception as e:
        print("Error during texture generation:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/executemeshroom', methods=['POST'])
def executemeshroom():
    try:
        Itemname = request.form.get('Itemname')  # Use get() to handle missing keys gracefully
        
#         prompt = request.form.get('prompt')
#         print(f"Received texture generation request for {itemname} with prompt: {prompt}")


        if not Itemname:
            return 'Itemname parameter is missing in the request', 400

        print("3D Model Generation Request received for: " + Itemname)

        # Create the output folder if it doesn't exist
        output_folder = os.path.join('TRELLIS Out Models', Itemname)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

#         output_folder = "TRELLIS Output Model\\" + Itemname  # Specify your desired output folder
        output_folder = os.path.join("TRELLIS Out Models", Itemname)

        item_folder = os.path.join('Received Images', Itemname, "Images")
        print(item_folder)
            # Load images
#         images = [Image.open(img) for img in img_files]
        images = [Image.open(os.path.join(item_folder, f)) for f in os.listdir(item_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


        # Run the pipeline
        outputs = pipeline.run_multi_image(
            images,
            seed=1,
            sparse_structure_sampler_params={
                "steps": 12,
                "cfg_strength": 7.5
            },
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 3
            },
        )

        # Create output directory
        output_dir = output_folder

#         output_dir = os.path.join(output_folder, f"{Itemname}")
        os.makedirs(output_dir, exist_ok=True)

        # Save mesh as GLB
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,
            texture_size=1024,
        )
        glb_path = os.path.join(output_dir, f"{Itemname}.glb")
        glb.export(glb_path)

        # Generate and save video
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]

        video_path = os.path.join(output_dir, "output.mp4")
        imageio.mimsave(video_path, video, fps=30)
        
        ply_path = os.path.join(output_dir, f"{Itemname}.ply")
        o3d.io.write_triangle_mesh(ply_path, outputs['mesh'][0])

        print(f"Processed {Itemname}: Mesh (.obj), GLB (.glb), and video saved in {output_dir}")
    #         
        print("Invoking TRELLIS")

        print("TRELLIS processing completed.")

        return 'TRELLIS Processing successfull'

    except Exception as e:
        print(f"Error during TRELLIS Processing: {str(e)}")
        return 'TRELLIS Processing failed'


@app.route("/getthumbnail", methods=["POST"])
def get_thumbnail():
    try:
        image_name = request.form.get('Itemname')
        print("Request image_name:", image_name)


        if image_name is None:
            return jsonify({"error": "Image name not provided"}), 400
        # data = request.json
        # image_name = data.get("imageName")
        print("Item name recieved " + image_name)
        image_extensions = ['.png', '.jpg', '.jpeg']  # Add more extensions as needed
        # image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']  # Add more extensions as needed

#         thumbnail_directory = "Received Images\\" + image_name + "\\"
        thumbnail_directory = os.path.join("Received Images", image_name)

        # Check if the image exists in the specified directory
        image_path = os.path.join(thumbnail_directory, image_name)

        for ext in image_extensions:
            thumbnail_path = os.path.join(image_path + ext)
            if os.path.exists(thumbnail_path):
                print("Found at " + thumbnail_path)
                return send_file(thumbnail_path, as_attachment=True)
                break  # Found the image, no need to check other extensions
            else:
                print("Seacrhing in thumbnail_path at " + thumbnail_path)
        # if os.path.isfile(image_path):
            # return send_file(image_path, as_attachment=True)
        # else:
        return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/getmodel", methods=["POST"])
def get_model():
    try:
        image_name = request.form.get('Itemname')
        print("Request image_name:", image_name)

        if image_name is None:
            return jsonify({"error": "Model name not provided"}), 400
        # data = request.json
        # image_name = data.get("imageName")
        print("Model name recieved " + image_name)
        image_extensions = ['.glb']  # Add more extensions as needed

#         thumbnail_directory = "Trellis Out Models\\" + image_name + "\\"
        thumbnail_directory = os.path.join("TRELLIS Out Models", image_name)

        # Check if the image exists in the specified directory
        image_path = os.path.join(thumbnail_directory, image_name)

        for ext in image_extensions:
            thumbnail_path = os.path.join(image_path + ext)
            if os.path.exists(thumbnail_path):
                print("Found at " + thumbnail_path)
                return send_file(thumbnail_path, as_attachment=True)
                break  # Found the image, no need to check other extensions
            else:
                print("Seacrhing in model_path at: " + thumbnail_path)
        # if os.path.isfile(image_path):
            # return send_file(image_path, as_attachment=True)
        # else:
        return jsonify({"error": "'{item_name}'Model not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Replace with the actual path to your SQLite database file
database_file_path = 'Server.s3db'

@app.route('/api/getDatabase', methods=['GET'])
def get_database():
    try:
        return send_file(database_file_path, as_attachment=True, download_name='Server.s3db')
    except FileNotFoundError:
        print("Database server file not found on Server!!")
        return "Database file not found", 404
    except Exception as e:
        return str(e), 500

thumbnails_dir = "Users\\wakee\\Extracted ZIP Contents\\"  # Replace with the actual path.





@app.route('/api/getMissingThumbnails', methods=['POST'])   #Ig not used in app
def get_missing_thumbnails():
    try:
        data = request.form.get('missingThumbnails')
        missing_thumbnails = data.split(',')

        found_thumbnails = {}

        # List of supported image file extensions
        image_extensions = ['.png', '.jpg', '.jpeg']  # Add more extensions as needed
        # image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']  # Add more extensions as needed

        for thumbnail_name in missing_thumbnails:
            found_thumbnail = None
            for ext in image_extensions:
                thumbnail_path = os.path.join(thumbnails_dir, thumbnail_name + ext)
                if os.path.exists(thumbnail_path):
                    with open(thumbnail_path, 'rb') as file:
                        found_thumbnail = file.read()
                        found_thumbnail_base64 = base64.b64encode(found_thumbnail).decode('utf-8')
                    break  # Found the image, no need to check other extensions

            if found_thumbnail:
                found_thumbnails[thumbnail_name] = found_thumbnail_base64
                print("Thumbnail found for " + thumbnail_name + " at " + thumbnail_path)

            else:
                print("Thumbnail not found for " + thumbnail_name)

        # Convert the found_thumbnails dictionary to JSON
        response_data = json.dumps(found_thumbnails)

        return response_data, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        # Log the exception traceback for debugging purposes
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/deletelisting', methods=['POST'])
def delete_listing():

    # if not os.path.exists(database_name):
        # return "Database not found", 404

    data = request.get_json()
    print("Received:", data)

    item_name = data.get('name')
    
    if not all([item_name]):
        return jsonify({"error": "Missing fields"}), 400

    print("delete_listing called for database:", item_name)

    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    # Check if the item exists
    cursor.execute("SELECT * FROM item WHERE Name = ?", (item_name,))
    result = cursor.fetchone()

    if result:
        cursor.execute("DELETE FROM item WHERE Name = ?", (item_name,))
        conn.commit()
        conn.close()
        return f"Item '{item_name}' deleted successfully", 200
    else:
        conn.close()
        return f"Item '{item_name}' not found", 404

# Create the user table if it doesn't exist
def create_user_table():
    conn = sqlite3.connect(database_file_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            Email TEXT NOT NULL UNIQUE,
            Password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


@app.route('/register', methods=['POST'])
def register():
    create_user_table()
    data = request.get_json()
    print("Received:", data)
    # return jsonify({"message": "User registered successfully!"}), 200
    name = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not all([name, email, password]):
        return jsonify({"error": "Missing fields"}), 400

    try:
        conn = sqlite3.connect(database_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user (Name, Email, Password)
            VALUES (?, ?, ?)
        ''', (name, email, password))
        conn.commit()
        conn.close()
        return jsonify({"message": "User registered successfully!"}), 200

    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already exists"}), 409

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Missing email or password'}), 400

    try:
        conn = sqlite3.connect(database_file_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user WHERE Email = ? AND Password = ?', (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            return jsonify({'message': 'Login successful'}), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401

    except Exception as e:
        print("Login Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    from pyngrok import ngrok

    # Optional: set your ngrok authtoken if needed
    # ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")

    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print(" * ngrok tunnel \"public_url\" -> \"http://127.0.0.1:5000\"")
    print(" * Access your API via:", public_url)

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)