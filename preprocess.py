import os
import skimage as sk
import random
from scipy import ndarray
from skimage import transform, util
import shutil


def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    #horizontal flip
    return image_array[:, ::-1]

# dictionary of the transformation functions

available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}


def copyDirectory(dir_to_copy, new_dir):
        if not os.path.exists(new_dir):
                shutil.copytree(dir_to_copy, new_dir)
        else:
                print("\nTried copying directory, but augmented directory already exists\n")
                
                
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
def fileCount(folder):
    "count the number of files in a directory"

    count = 0

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if os.path.isfile(path):
            count += 1
        elif os.path.isfolder(path):
            count += fileCount(path)    

    return count

def distribute_images_in_folder(folder_path, folder_name,training_set, validation_set, test_set):
    
    number_of_images = fileCount(folder_path)
    # where training set stops
    training_index = int(number_of_images * 0.8)
    # where validation set stops
    validation_index = int(number_of_images* 0.9)

    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_index = 0
    # making training set
    for image in images[:training_index]:
            read_image = sk.io.imread(image)
            new_file_path = '%s/%s/image_%s.jpg' % (training_set, folder_name, image_index)
            sk.io.imsave(new_file_path, read_image)
            image_index += 1
    
    # making validation set
    for image in images[training_index:validation_index]:
        read_image = sk.io.imread(image)
        new_file_path = '%s/%s/image_%s.jpg' % (validation_set, folder_name, image_index)
        sk.io.imsave(new_file_path, read_image)
        image_index += 1
        
    # make test setlocation
    for image in images[validation_index:]:
        read_image = sk.io.imread(image)
        new_file_path = '%s/%s/image_%s.jpg' % (test_set, folder_name, image_index)
        sk.io.imsave(new_file_path, read_image)
        image_index += 1


    
def augment_folder(folder_path, desired_size):
    
    # random num of transformations to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))
    
    #num_transformatinos = 0
    transformation_image = None
    
    # loop on all files of the folder and build a list of files paths
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    num_files = fileCount(folder_path)
    num_generated_files = 0
    
    while num_files < desired_size:
        # random image from the folder
        image_path = random.choice(images)
        
        # read image as an two dimensional array of pixels
        image_to_transform = sk.io.imread(image_path)
        
        
        # random num of transformations to apply
        #num_transformations_to_apply = random.randint(1, len(available_transformations))
        #num_transformations = 0
        transformed_image = horizontal_flip(image_to_transform)
        
        #while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
         #   key = random.choice(list(available_transformations))
         #   transformed_image = available_transformations[key](image_to_transform)
         #   num_transformations += 1
        
        # define a name for new file
        new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)
        
        #write image to the disk
        sk.io.imsave(new_file_path, transformed_image)
        num_files += 1
        num_generated_files += 1
        
def sample_folder(folder_path, desired_size):
    
    num_files = fileCount(folder_path)
    
    while num_files > desired_size:
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        image_path = random.choice(images)
        os.remove(image_path)
        num_files -= 1
            
def sample_set(training_set, desired_size):
    for root, dirs, files in os.walk(training_set):
        for name in dirs:
            print("Resizing class: " + name)
            folder_path = training_set + "/" + name
            if fileCount(folder_path) < desired_size:
                augment_folder(folder_path, desired_size)
            else:
                sample_folder(folder_path, desired_size)


def preprocess(datasource, desired_size):                       
    
    new_dataset_path = os.getcwd() + "/Data/preprocessed_imgs"
    make_dir(new_dataset_path)
    
    #make training set
    training_set = new_dataset_path + "/training_set"
    make_dir(training_set)
    
    #make validation set
    validation_set = new_dataset_path + "/validation_set"
    make_dir(validation_set)
    
    #make test set
    test_set = new_dataset_path + "/test_set"
    
    class_names_path = "Ex1_selected_categories"
    with open(class_names_path) as f:
        lines = f.read().splitlines()
        
    for line in lines:
        for root, dirs, files in os.walk(datasource):
            for name in dirs:
                if line[1:] == name:
                    print("Now extracting class: " + name):
                    if not os.path.exists(new_dataset_path + "/" + name):
                        make_dir(training_set + "/" + name)
                        make_dir(validation_set + "/" + name)
                        make_dir(test_set + "/" + name)
                        distribute_images_in_folder(datasource + "/" + name, name,training_set, validation_set, test_set):
    
    
    print("Finished making train/val/test sets")
    
    sample_set(training_set, desired_size)
    


if __name__ == "__main__":
    preprocess("/mnt/disks/sdbt/zooscannet/ZooScanSet/imgs", 7500)

                             
