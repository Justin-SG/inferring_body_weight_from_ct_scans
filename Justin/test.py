from totalsegmentator.python_api import totalsegmentator

input_path = "C:/Users/schoe/Desktop/inferring_body_weight_from_ct_scans/0_Data_Understanding/CT_example/1001EE27"
output_path = "test2"

if __name__ == "__main__":
    totalsegmentator(input_path, output_path, fast=True, preview=True)