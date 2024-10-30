from infer_endoscopy_image_detection import infer_endoscopy_image_detection
from infer_lesion_detection import infer_lesion_detection


async def infer(img_path: str):
    print(f"Running inference on: {img_path}")

    endoscopy_img_list_path = infer_endoscopy_image_detection(img_path)
    if endoscopy_img_list_path and len(endoscopy_img_list_path) > 0:
        lesion_list_path = infer_lesion_detection(endoscopy_img_list_path)
    else:
        lesion_list_path = infer_lesion_detection([])
    return {
        "img_path": img_path,
        "endoscopy_img_list_path": endoscopy_img_list_path,
        "lesion_list_path": lesion_list_path
    }