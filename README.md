# Testing Instructions

To test the model, please follow the steps below:

1.  **Configure the Test Dataset Path:**
    Open the configuration file located at `codes/config/denoising-sde/options/test/refusion.yml`.
    Find the line containing `dataroot_LQ:` and paste the root directory of your test dataset after it. For example:

    ```yaml
    dataroot_LQ: /path/to/your/test/datasets
    ```

2.  **Run the Test Script:**
    Execute the following command in your terminal:

    ```bash
    sh test.sh
    ```

3.  **Output Location:**
    After the testing process is complete, the predicted (denoised) images will be saved in the following directory:

    ```
    results/denoising-sde/refusion/NTIRETEST
    ```