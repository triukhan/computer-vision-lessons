import cv2
import pytesseract
from PIL import Image
from easyocr import Reader
import boto3


image_path = 'opencv_lessons/data/text_detection/pexels-thngocbich-760724.jpg'
reader = Reader(['en'])

access_key = None
secret_access_key = None

textract_client = boto3.client('textract',
                               aws_access_key_id=access_key,
                               aws_secret_access_key=secret_access_key,
                               region_name='us-east-1')


def read_text_tesseract(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cfg = '--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=cfg)
    return text


def read_text_easyocr(img_path):
    text = ''

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    results = reader.readtext(thresh)
    for result in results:
        text = text + result[1] + ' '

    text = text[:-1]
    return text


def read_text_textract(img_path):
    with open(img_path, 'rb') as im:
        response = textract_client.detect_document_text(Document={'Bytes':im.read()})

    text = ''
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            text = text + item['Text'] + ' '

    text = text[:-1]
    return text


result = read_text_easyocr(image_path)
print(result)
