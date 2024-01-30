## About
Jupyter notebook and python scripts used for blog post https://paul.mou.dev/posts/2023-12-31-listening-with-llm/

## Warning
Note the jupyterbook scripts and path references are custom to my local setup and haven't been cleaned up. These files are for reference and won't run as-is.


## Update 2024-01-30
Added `proj` module checkpoint saved via

```
torch.save(
    model.audio_encoder.proj.state_dict(),
    <OUT>,
)
```

see https://github.com/moomou/listening-with-llm/blob/master/model_manual_save3.pth
