import os, cv2, math
from PIL import Image, ImageEnhance, ImageFilter
from PIL import ImageOps 
import numpy as np
import matplotlib.pyplot as plt
from retinaface import RetinaFace

def augment_image(image_path, name_folder, person_folder_path, people_folder):

    print("here", image_path ,"\n", name_folder, "\n",person_folder_path)
    print("People Folder", people_folder)

    output_folder = os.path.join(people_folder, name_folder)
    print(output_folder)
    save_directory = output_folder

    # Load the image
    image = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    # save_path = os.path.join(name_folder, image_name)
    # print(save_path, "Save_path", save_directory)
    # output_folder = name_folder
    # os.makedirs(output_folder, exist_ok=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    #Let's change the directory in running time...
    new_dir = output_folder
    os.chdir(new_dir)

    # Apply various image transformations
    # 1. Increase image brightness
    enhancer = ImageEnhance.Brightness(image)
    bright_image = enhancer.enhance(1.5)
    bright_image.save(f"{image_name}_brightened.png")

    # 2. Decrease image brightness
    dark_image = enhancer.enhance(0.5)
    dark_image.save(f"{image_name}_dark.png")

    # 3. Increase image contrast
    enhancer = ImageEnhance.Contrast(image)
    high_contrast_image = enhancer.enhance(2.0)
    high_contrast_image.save(f"{image_name}_high_contrast.png")

    # 4. Decrease image contrast
    low_contrast_image = enhancer.enhance(0.5)
    low_contrast_image.save(f"{image_name}_low_contrast.png")

    # 5. Adjust image saturation
    enhancer = ImageEnhance.Color(image)
    saturated_image = enhancer.enhance(1.5)
    saturated_image.save(f"{image_name}_saturated.png")

    # 6. Desaturate image
    desaturated_image = enhancer.enhance(0.5)
    desaturated_image.save(f"{image_name}_desaturated.png")

    # 7. Enhance image sharpness
    enhancer = ImageEnhance.Sharpness(image)
    sharpened_image = enhancer.enhance(2.0)
    sharpened_image.save(f"{image_name}_sharpened.png")

    # 8. Reduce image sharpness
    blurred_image = enhancer.enhance(0.5)
    blurred_image.save(f"{image_name}_blurred.png")

    # Convert the image to RGB mode
    image = image.convert("RGB")

    # 9. Apply image posterization
    posterized_image = ImageOps.posterize(image, 4)
    posterized_image.save(f"{image_name}_posterized.png")

    # 10. Apply image equalization
    equalized_image = ImageOps.equalize(image)
    equalized_image.save(f"{image_name}_equalized.png")

    #11. Applying Clahe stuff to get high contrast images 

    if not isinstance(image, np.ndarray):
        print("Convert if the image is not a NumPy array")
        image = np.array(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)#............
    lab_planes = cv2.split(lab)#............
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))#............
    print(len(lab_planes),lab_planes)
    a = clahe.apply(lab_planes[0])#............
    lab_planes = (a,) + lab_planes[1:]#............
    lab = cv2.merge(lab_planes)#............
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)#............
    clahe_bgr_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB) #............
    cv2.imwrite(f"{image_name}_clahe.png", clahe_bgr_rgb)

    
def preprocess_image(image_path, current_folder_name, person_folder):

    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # Normalize the image
    normalized_image = cv2.normalize(image_rgb, None, alpha=0, beta=300, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Performing face alignment
    resp = RetinaFace.detect_faces(image_path)
    # print(resp)

    try :
        x1, y1 = resp["face_1"]["landmarks"]["right_eye"]
        x2, y2 = resp["face_1"]["landmarks"]["left_eye"]

        a = abs(y1 - y2)
        b = abs(x1 - x2)
        c = math.sqrt(a*a + b*b)

        cos_alpha = (b*b + c*c - a*a) / (2*b*c)
        alpha = np.arccos(cos_alpha)
        alpha = (alpha * 180) / math.pi
        # alpha

        aligned_image = Image.fromarray(image)
        aligned_image = np.array(aligned_image.rotate(360-alpha))

    except Exception as tuple_error :
        print("Skipping alignment for this image due to the following error :\n", str(tuple_error))
        aligned_image = normalized_image



    # # Path to the original image
    # original_image_path = "path/to/original_image.png"

    # # Path to the generated aligned image
    # aligned_image_path = "path/to/aligned_image.png"

    # Get the original image's name without the file extension
    original_image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # current_directory = os.getcwd()
    # current_folder_name = os.path.basename(current_directory)
    
    # Create a new folder with suffic "_processed_images" in the current directory
    output_folder = person_folder + "_processed_images"
    os.makedirs(output_folder, exist_ok=True)

    # Rename the aligned image to the original image's name + "processed.png"
    processed_image_name = original_image_name + "_processed.png"
    processed_image_path = os.path.join(output_folder, processed_image_name)

    # Load the aligned image
    # aligned_image = cv2.imread(aligned_image_path)
    # Since it's already loaded with cv2 isn't it 

    # Save the aligned image with the processed name in the processed_images folder
    cv2.imwrite(processed_image_path, aligned_image)


    # # Extract the file name and extension
    # file_name, file_extension = os.path.splitext(image_path)

    # # Create the new file name with the "_processed.png" suffix
    # new_file_name = file_name + '_processed.png'

    # # cv2.imwrite(new_file_name, aligned_image)

    # # Specify the path to the desired destination directory
    
    # current_directory = os.getcwd()
    # # destination_directory = current_directory + "/processed_folder"  # Replace with the path to the desired destination directory

    # new_folder = 'processed_folder'
    # os.makedirs(new_folder, exist_ok=True)

    # # Create the complete destination file path
    # destination_file = os.path.join(new_folder, new_file_name)
    
    # cv2.imwrite(destination_file, aligned_image)

    # # aligned_image = Image.fromarray(aligned_image)
    # # # Save the new image to the destination directory
    # # aligned_image.save(destination_file)


# wait
"""
# Example usage
input_image_path = "/Users/akshatkalra/Desktop/Face Recognition Project/Photos/Kyle Fenole/fenolekyle_41137_7029521_fenole_0425-2.jpg"
output_directory = "augmentation"

# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print("Current directory:", current_directory)

# augment_image(input_image_path, output_directory)


preprocess_image(input_image_path)

"""

# Folder path containing the list of people names
people_folder = "/Users/akshatkalra/Desktop/Face Recognition Project/Photos/Processed Dataset"
supported_extensions = [".jpeg", ".png", ".jpg"]

# current_folder_name = "v"
# person_folder = "a"
# image_path = "/Users/akshatkalra/Desktop/Face Recognition Project/Photos/Chakrapani Chitnis/chitnischakrapani_56622_7031903_image_0501.jpg"
# preprocess_image(image_path, current_folder_name, person_folder)

"""
# # For Pre-Processing Images
for person_name in os.listdir(people_folder):
    person_folder = os.path.join(people_folder, person_name)

    print(person_folder)
    
    # Check if the item in the people folder is a directory
    if os.path.isdir(person_folder):
        current_folder_name = os.path.basename(person_folder)
        # Iterate over each image in the person folder
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            
            print(image_path)
            print(current_folder_name)
            n = len("_processed_images")
            name_folder = current_folder_name[ : -n]
            print(name_folder)
            # # Check if the item in the person folder is a file
            # if os.path.isfile(image_path) and any(image_path.lower().endswith(ext) for ext in supported_extensions) :
            #     # Process the image using the processing function
            #     preprocess_image(image_path, current_folder_name, person_folder)

"""

#For performing augmentations 

for person_name in os.listdir(people_folder):
    person_folder = os.path.join(people_folder, person_name)

    print(person_folder)
    
    # Check if the item in the people folder is a directory
    if os.path.isdir(person_folder):
        current_folder_name = os.path.basename(person_folder)
        # Iterate over each image in the person folder
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            
            print(image_path)
            print(current_folder_name)
            n = len("_processed_images")
            name_folder = current_folder_name[ : -n]
            print(name_folder)
            augment_image(image_path, name_folder, person_folder, people_folder)


# augment_image(image_path, name_folder, person_folder_path)