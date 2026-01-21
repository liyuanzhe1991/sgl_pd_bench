export SGLANG_HEALTH_CHECK_TIMEOUT=240
python -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://10.6.131.23:30500 \
    --decode http://10.6.131.20:30001 \
    --host 0.0.0.0 --port 8000
