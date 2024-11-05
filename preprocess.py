import os
import shutil

class Preprocess:
    def rename_files(self, folder_path):
        jpg_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        xml_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xml')])


        for i, (jpg, xml) in enumerate(zip(jpg_files, xml_files), start=1):
            new_name = f"{i:05d}"
            
            old_jpg_path = os.path.join(folder_path, jpg)
            old_xml_path = os.path.join(folder_path, xml)
            new_jpg_path = os.path.join(folder_path, f"{new_name}.jpg")
            new_xml_path = os.path.join(folder_path, f"{new_name}.xml")
            
            os.rename(old_jpg_path, new_jpg_path)
            os.rename(old_xml_path, new_xml_path)
    
    def move_files(self, folder_path):

        jpg_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        xml_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xml')])

        for file in jpg_files:
            shutil.move(os.path.join(folder_path, file) , os.path.join(folder_path, "images", file))
        
        for file in xml_files:
            shutil.move(os.path.join(folder_path, file) , os.path.join(folder_path, "annots", file))


preprocess = Preprocess()
preprocess.rename_files("train")
preprocess.move_files("train")




