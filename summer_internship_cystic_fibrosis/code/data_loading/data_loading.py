import shutil
import os
from PyPDF2 import PdfReader
import pandas as pd
import glob

# We have 2 sources of dataset to load:
#   - The 3D csv files for specific ventilation (in peak inspiration)
#   - The pdf files that we can create report_summary csv file

# The general report summary will have the `ScanName`, `Date Prepared` 5 parameters: VDP, MSV, TV, VH, VHSS, VHLS 
# Additional data like Genotype, Disease type, Weight, Sex, etc. will be added according to the dataset given



def get_all_pdfs(folder_path):
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        pdf_files.extend(glob.glob(os.path.join(root, '*.pdf')))
    return pdf_files

# Function to remove all non-norm files from the list
def remove_non_norm_files(files_list):
    new_files_list = []
    for file in files_list:
        if "non-norm" not in file:
            new_files_list.append(file)
    return new_files_list

# Function to generate the report summary created from the pdf files
# The output will be Pandas Dataframe
def create_report_summary(source_folder):
    pdf_files = [f for f in get_all_pdfs(folder_path=source_folder)]
    data = [
        ["ScanName", "DatePrepared", "VDP(%)", "MSV(mL/mL)", "TV(L)", "VH(%)", "VHSS(%)", "VHLS(%)", "FileName"]
    ]
    pdf_files = remove_non_norm_files(pdf_files)
    for pdf_file in pdf_files:

        # Render the pages to extract the text
        reader = PdfReader(pdf_file)
        page = reader.pages[0]
        text = page.extract_text().split("\n")

        # Extracting information from the text
        scan_name = text[1].split()[-1]
        date_prepared = text[2].split()[2] + "-" + text[2].split()[3]
        vdp = float(text[3].split()[-1])
        msv = float(text[4].split()[-1])
        tv = float(text[5].split()[-1])
        vh = float(text[6].split()[-1])
        vhss = float(text[7].split()[-1])
        vhls = float(text[8].split()[-1])

        # if MAL_ in scan_name, replace it with MAL-
        if "MAL_" in scan_name:
            scan_name = scan_name.replace("MAL_", "MAL-")

        #We will add the file name for further extraction later
        row = [scan_name, date_prepared, vdp, msv, tv, vh, vhss, vhls, pdf_file.split("/")[-1]]
        data.append(row)

    # Create the Pandas DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])
    
    return df

# Function to copy one file from one path to the other
def copy_csv(source_path, destination_path):
    try:
        shutil.copy(source_path, destination_path)
    except Exception as e:
        print(f"Error: {e}")

#Function to copy all the csv files from the source folder to the destination folder
def copy_3d_csvs(source_folder, destination_folder):
    csv_files = [f for f in os.listdir(source_folder) if f.endswith(".csv")]
    for csv_file in csv_files:
        os.makedirs(destination_folder, exist_ok=True)
        copy_csv(source_folder+csv_file,destination_folder+csv_file)


#Function to copy a folder to the destination folder
def copy_folder(source_folder, destination_folder):
    try:
        # Copy the entire contents of the source folder to the destination folder
        shutil.copytree(source_folder, destination_folder)
        print(f"Folder '{source_folder}' successfully copied to '{destination_folder}'.")
    except shutil.Error as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
