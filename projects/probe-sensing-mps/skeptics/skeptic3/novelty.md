# Novelty Check

The finding is novel. Prior literature, such as FloatDoor (arXiv 2606.19535), trained platform-detection models strictly on NVIDIA chips. Apple MPS (M1/M2/M4) platforms were only measured in the FloatDoor appendix for baseline differences in the residual streams but no probe-sensing detection models were trained. Furthermore, Sauers' watching project only tested CUDA implementations. Establishing that this leakage transfers over to PyTorch compilation on MPS but vanishes when running via MLX compilation is a new observation.
