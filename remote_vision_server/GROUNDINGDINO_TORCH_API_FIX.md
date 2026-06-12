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

## How To Recognize A Skipped Extension Build

This log means the extension was not built:

```text
Building editable for groundingdino ... done
Created wheel for groundingdino: groundingdino-0.1.0-0.editable-py3-none-any.whl
Successfully installed groundingdino-0.1.0
```

The important clue is:

```text
py3-none-any.whl
```

and there is no `running build_ext` / no `_C*.so` file. In that case:

```bash
python -c "import groundingdino._C"
```

will still fail. This usually means `setup.py` is still patched to skip extension compilation, or `ext_modules` is empty.

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

If the GroundingDINO checkout still has its `.git` directory, the simplest restore is:

```bash
cd /home/ma-user/work/zhangWei/mtu3d/data/trans/agent_based_method/third_party/GroundingDINO
git checkout -- setup.py
```

Then confirm it is not skipping extensions:

```bash
grep -n "def get_extensions\\|return None\\|ext_modules" setup.py
```

`return None` must not appear inside `get_extensions()`.

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
find groundingdino -name "_C*.so" -print
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
