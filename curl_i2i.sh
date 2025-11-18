curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d'{
    "resize_height": 1536,
    "resize_width": 1536,
    "input_image": "/app/OMGSR/my_tests/image_beauty.png",
    "output_dir": "experiments_omgsr_f",
    "process_size": 1024,
    "upscale": 4,
    "mid_timestep": 244,
    "align_method": "adain",
    "guidance_scale": 1.0
  }'