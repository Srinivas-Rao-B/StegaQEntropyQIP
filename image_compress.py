from PIL import Image
import io
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\Users\HP\Desktop\Stego"

INPUT_IMAGE = os.path.join(
    BASE_DIR,
    "images.jpg"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "demo_image_test"
)

TARGET_BITS = 25_000

IMAGE_SIZE = 128


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_256 = os.path.join(
    OUTPUT_DIR,
    "original_256.png"
)

OUTPUT_128 = os.path.join(
    OUTPUT_DIR,
    "original_128.png"
)

OUTPUT_COMPRESSED = os.path.join(
    OUTPUT_DIR,
    "compressed_128.jpg"
)

OUTPUT_BINARY = os.path.join(
    OUTPUT_DIR,
    "compressed_128.bin"
)

OUTPUT_BINARY_TEXT = os.path.join(
    OUTPUT_DIR,
    "compressed_128.txt"
)

CNN_DEGRADED = os.path.join(
    OUTPUT_DIR,
    "cnn_degraded_128.png"
)

CNN_ENHANCED = os.path.join(
    OUTPUT_DIR,
    "cnn_restored_128.png"
)

CNN_MODEL = os.path.join(
    OUTPUT_DIR,
    "cnn_model_128.pth"
)


# ============================================================
# CNN SETTINGS
# ============================================================

CNN_EPOCHS = 500

CNN_LEARNING_RATE = 0.001

CNN_FEATURES = 32


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_IMAGE):

    raise FileNotFoundError(
        f"\nOriginal image not found:\n{INPUT_IMAGE}"
    )


# ============================================================
# LOAD ORIGINAL 256×256
# ============================================================

image_256 = Image.open(
    INPUT_IMAGE
).convert("L")


image_256 = image_256.resize(
    (256, 256),
    Image.Resampling.LANCZOS
)


# ============================================================
# SAVE ORIGINAL 256×256
# ============================================================

image_256.save(
    OUTPUT_256,
    format="PNG"
)


# ============================================================
# CONVERT 256 → 128
# ============================================================

image_128 = image_256.resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    Image.Resampling.LANCZOS
)


# ============================================================
# SAVE ORIGINAL 128×128
# ============================================================

image_128.save(
    OUTPUT_128,
    format="PNG"
)


# ============================================================
# JPEG COMPRESSION FUNCTION
# ============================================================

def compress_jpeg(
    image,
    quality
):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=False
    )

    return buffer.getvalue()


# ============================================================
# SEARCH FOR HIGHEST JPEG QUALITY ≤25,000 BITS
# ============================================================

best_data = None

best_quality = None


print()
print("==============================================")
print("128×128 IMAGE COMPRESSION")
print("==============================================")


print()
print("Input:")
print(INPUT_IMAGE)


print()
print("Original resolution:")
print("256 × 256")


print()
print("Compression resolution:")
print("128 × 128")


print()
print("Target maximum:")
print(f"{TARGET_BITS:,} bits")


print()
print("Target maximum bytes:")
print(f"{TARGET_BITS // 8:,} bytes")


print()
print("Searching for highest JPEG quality...")


for quality in range(
    100,
    0,
    -1
):

    data = compress_jpeg(
        image_128,
        quality
    )

    size_bits = len(data) * 8


    if size_bits <= TARGET_BITS:

        best_data = data

        best_quality = quality

        break


# ============================================================
# CHECK RESULT
# ============================================================

if best_data is None:

    raise RuntimeError(
        "Could not compress image below 25,000 bits."
    )


# ============================================================
# SAVE COMPRESSED JPEG
# ============================================================

with open(
    OUTPUT_COMPRESSED,
    "wb"
) as f:

    f.write(
        best_data
    )


# ============================================================
# SAVE COMPRESSED BYTES
# ============================================================

with open(
    OUTPUT_BINARY,
    "wb"
) as f:

    f.write(
        best_data
    )


# ============================================================
# CONVERT JPEG BYTES → 0/1
# ============================================================

binary_data = "".join(
    format(
        byte,
        "08b"
    )
    for byte in best_data
)


# ============================================================
# SAVE 0/1 PAYLOAD
# ============================================================

with open(
    OUTPUT_BINARY_TEXT,
    "w"
) as f:

    f.write(
        binary_data
    )


# ============================================================
# STATISTICS
# ============================================================

raw_bits_256 = (
    256 *
    256 *
    8
)


raw_bits_128 = (
    128 *
    128 *
    8
)


compressed_bytes = len(
    best_data
)


compressed_bits = (
    compressed_bytes *
    8
)


compression_ratio = (
    raw_bits_128 /
    compressed_bits
)


reduction = (
    1 -
    compressed_bits /
    raw_bits_128
) * 100


# ============================================================
# COMPRESSION RESULT
# ============================================================

print()
print("==============================================")
print("COMPRESSION RESULT")
print("==============================================")


print()
print("Original:")
print("256 × 256")


print()
print("Compressed:")
print("128 × 128")


print()
print("Color:")
print("Grayscale")


print()
print("Raw 256×256:")
print(f"{raw_bits_256:,} bits")


print()
print("Raw 128×128:")
print(f"{raw_bits_128:,} bits")


print()
print("JPEG quality:")
print(best_quality)


print()
print("Compressed size:")
print(f"{compressed_bytes:,} bytes")


print()
print("Compressed bits:")
print(f"{compressed_bits:,} bits")


print()
print("Compression ratio:")
print(f"{compression_ratio:.2f}:1")


print()
print("Size reduction from 128×128:")
print(f"{reduction:.2f}%")


# ============================================================
# FILES SAVED
# ============================================================

print()
print("==============================================")
print("FILES SAVED")
print("==============================================")


print()
print("Original 256×256:")
print(OUTPUT_256)


print()
print("Original 128×128:")
print(OUTPUT_128)


print()
print("Compressed JPEG:")
print(OUTPUT_COMPRESSED)


print()
print("Compressed binary:")
print(OUTPUT_BINARY)


print()
print("0/1 payload:")
print(OUTPUT_BINARY_TEXT)


print()
print("==============================================")
print("BINARY PAYLOAD")
print("==============================================")


print()
print("First 128 bits:")
print(binary_data[:128])


print()
print("Total binary length:")
print(f"{len(binary_data):,} bits")


print()
print("==============================================")


if compressed_bits <= TARGET_BITS:

    print(
        "STATUS: ✓ UNDER 25,000 BITS"
    )

else:

    print(
        "STATUS: ✗ OVER 25,000 BITS"
    )


print("==============================================")


# ============================================================
# CNN RESTORATION
# ============================================================


# ============================================================
# SIMPLE CNN
# ============================================================

class RestorationCNN(
    nn.Module
):


    def __init__(
        self
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Conv2d(
                1,
                CNN_FEATURES,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),


            nn.Conv2d(
                CNN_FEATURES,
                CNN_FEATURES,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),


            nn.Conv2d(
                CNN_FEATURES,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),


            nn.Conv2d(
                16,
                1,
                kernel_size=3,
                padding=1
            )
        )


    def forward(
        self,
        x
    ):

        return self.network(x)


# ============================================================
# LOAD COMPRESSED IMAGE
# ============================================================

compressed_image = Image.open(
    OUTPUT_COMPRESSED
).convert("L")


compressed_image = compressed_image.resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    Image.Resampling.LANCZOS
)


# ============================================================
# SAVE DEGRADED IMAGE
# ============================================================

compressed_image.save(
    CNN_DEGRADED
)


# ============================================================
# IMAGE → TENSOR
# ============================================================

def image_to_tensor(
    image
):

    array = np.asarray(
        image,
        dtype=np.float32
    )


    array = (
        array /
        255.0
    )


    tensor = torch.from_numpy(
        array
    )


    tensor = tensor.unsqueeze(
        0
    )


    tensor = tensor.unsqueeze(
        0
    )


    return tensor


# ============================================================
# CREATE TENSORS
# ============================================================

input_tensor = image_to_tensor(
    compressed_image
).to(DEVICE)


target_tensor = image_to_tensor(
    image_128
).to(DEVICE)


# ============================================================
# CREATE CNN
# ============================================================

model = RestorationCNN().to(
    DEVICE
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=CNN_LEARNING_RATE
)


# ============================================================
# LOSS
# ============================================================

loss_function = nn.MSELoss()


# ============================================================
# TRAINING
# ============================================================

print()
print("==============================================")
print("CNN IMAGE RESTORATION")
print("==============================================")


print()
print("Device:")
print(DEVICE)


print()
print("Resolution:")
print("128 × 128")


print()
print("Training epochs:")
print(CNN_EPOCHS)


print()
print("Learning rate:")
print(CNN_LEARNING_RATE)


print()
print("Training...")


for epoch in range(
    CNN_EPOCHS
):


    model.train()


    optimizer.zero_grad()


    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    output = model(
        input_tensor
    )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss = loss_function(
        output,
        target_tensor
    )


    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------

    loss.backward()


    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        epoch == 0
        or
        (epoch + 1) % 25 == 0
    ):

        print(
            f"Epoch "
            f"{epoch + 1:4d}/{CNN_EPOCHS} "
            f"Loss: "
            f"{loss.item():.8f}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    CNN_MODEL
)


# ============================================================
# RESTORE IMAGE
# ============================================================

model.eval()


with torch.no_grad():

    restored = model(
        input_tensor
    )


    restored = torch.clamp(
        restored,
        0.0,
        1.0
    )


# ============================================================
# TENSOR → IMAGE
# ============================================================

restored = restored.squeeze()


restored = restored.cpu().numpy()


restored = (
    restored *
    255.0
).clip(
    0,
    255
).astype(
    np.uint8
)


restored_image = Image.fromarray(
    restored,
    mode="L"
)


# ============================================================
# SAVE RESTORED IMAGE
# ============================================================

restored_image.save(
    CNN_ENHANCED
)


# ============================================================
# NUMPY ARRAYS
# ============================================================

original_np = np.asarray(
    image_128,
    dtype=np.float32
)


compressed_np = np.asarray(
    compressed_image,
    dtype=np.float32
)


restored_np = np.asarray(
    restored_image,
    dtype=np.float32
)


# ============================================================
# MSE
# ============================================================

degraded_mse = mean_squared_error(
    original_np,
    compressed_np
)


cnn_mse = mean_squared_error(
    original_np,
    restored_np
)


# ============================================================
# MAE
# ============================================================

degraded_mae = np.mean(
    np.abs(
        original_np -
        compressed_np
    )
)


cnn_mae = np.mean(
    np.abs(
        original_np -
        restored_np
    )
)


# ============================================================
# PSNR
# ============================================================

degraded_psnr = peak_signal_noise_ratio(
    original_np,
    compressed_np,
    data_range=255
)


cnn_psnr = peak_signal_noise_ratio(
    original_np,
    restored_np,
    data_range=255
)


# ============================================================
# SSIM
# ============================================================

degraded_ssim = structural_similarity(
    original_np,
    compressed_np,
    data_range=255
)


cnn_ssim = structural_similarity(
    original_np,
    restored_np,
    data_range=255
)


# ============================================================
# IMPROVEMENT
# ============================================================

psnr_improvement = (
    cnn_psnr -
    degraded_psnr
)


ssim_improvement = (
    cnn_ssim -
    degraded_ssim
)


mse_reduction = (
    (
        degraded_mse -
        cnn_mse
    )
    /
    degraded_mse
) * 100


mae_reduction = (
    (
        degraded_mae -
        cnn_mae
    )
    /
    degraded_mae
) * 100


# ============================================================
# QUALITY COMPARISON
# ============================================================

print()
print("============================================================")
print("              128×128 CNN QUALITY COMPARISON")
print("============================================================")


print()


print(
    f"{'METRIC':<25}"
    f"{'DEGRADED':>15}"
    f"{'CNN':>18}"
)


print(
    "-" * 60
)


print(
    f"{'MSE':<25}"
    f"{degraded_mse:>15.4f}"
    f"{cnn_mse:>18.4f}"
)


print(
    f"{'MAE':<25}"
    f"{degraded_mae:>15.4f}"
    f"{cnn_mae:>18.4f}"
)


print(
    f"{'PSNR (dB)':<25}"
    f"{degraded_psnr:>15.4f}"
    f"{cnn_psnr:>18.4f}"
)


print(
    f"{'SSIM':<25}"
    f"{degraded_ssim:>15.6f}"
    f"{cnn_ssim:>18.6f}"
)


print(
    "-" * 60
)


# ============================================================
# CNN IMPROVEMENT
# ============================================================

print()
print("CNN IMPROVEMENT")
print("-" * 60)


print(
    f"PSNR improvement : "
    f"{psnr_improvement:+.4f} dB"
)


print(
    f"SSIM improvement : "
    f"{ssim_improvement:+.6f}"
)


print(
    f"MSE reduction    : "
    f"{mse_reduction:+.2f}%"
)


print(
    f"MAE reduction    : "
    f"{mae_reduction:+.2f}%"
)


# ============================================================
# FINAL FILES
# ============================================================

print()
print("============================================================")
print("FILES")
print("============================================================")


print()
print("Original 256×256:")
print(OUTPUT_256)


print()
print("Original 128×128:")
print(OUTPUT_128)


print()
print("Compressed:")
print(OUTPUT_COMPRESSED)


print()
print("0/1 payload:")
print(OUTPUT_BINARY_TEXT)


print()
print("CNN degraded:")
print(CNN_DEGRADED)


print()
print("CNN restored:")
print(CNN_ENHANCED)


print()
print("CNN model:")
print(CNN_MODEL)


print()
print("Resolution:")
print(restored_image.size)


print()
print("============================================================")
print("✓ 128×128 CNN TEST COMPLETE")
print("============================================================")