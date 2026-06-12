# GroundingDINO `_C` Extension Fix for New PyTorch/CUDA

## Do Not Skip The CUDA Extension

Do not modify `third_party/GroundingDINO/setup.py` like this:

```python
def get_extensions():
    return None
```

The remote vision service uses GroundingDINO's `ms_deform_attn` inference path, which depends on:

```python
groundingdino._C
```

Skipping the extension only postpones the problem. Installation may appear to pass, but inference fails with:

```text
name '_C' is not defined
```

## Why Compilation Failed

Your compile log shows:

```text
no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType" exists
```

This comes from old GroundingDINO CUDA source using older PyTorch C++ APIs such as:

```cpp
AT_DISPATCH_FLOATING_TYPES(value.type(), ...)
value.data<scalar_t>()
```

Newer PyTorch expects APIs like:

```cpp
AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), ...)
value.data_ptr<scalar_t>()
```

## Fix Steps On The Server

First restore the original `third_party/GroundingDINO/setup.py`. If you changed `get_extensions()` to `return None`, undo that change.

Then run the patch script from the project root:

```bash
conda activate pytorch_cuda12.9
cd /home/ma-user/work/zhangWei/mtu3d/data/trans/agent_based_method

python remote_vision_server/patch_groundingdino_torch_api.py \
  --groundingdino-root third_party/GroundingDINO
```

Rebuild GroundingDINO:

```bash
cd /home/ma-user/work/zhangWei/mtu3d/data/trans/agent_based_method/third_party/GroundingDINO

export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export FORCE_CUDA=1

pip install -e . --no-build-isolation -v
```

Verify:

```bash
python -c "import groundingdino._C; print('GroundingDINO ops ok')"
```

Then check the full remote vision environment:

```bash
cd /home/ma-user/work/zhangWei/mtu3d/data/trans/agent_based_method
python remote_vision_server/check_server_env.py
```

Restart the server:

```bash
bash remote_vision_server/run_server.sh
```

## If `which nvcc` Is Empty

Install CUDA compiler tools:

```bash
conda install -c nvidia cuda-nvcc cuda-toolkit -y
```

Then repeat the rebuild step.
