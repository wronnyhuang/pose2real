[ -f /root/FIX1 ] && { cat /root/q.log | tail -3; exit 0; }
touch /root/FIX1
printf '%s\n' 't_nohands 45 --bs 4 --ckpt --hands none --gan_start 100000' 't_inj_controlnet 45 --bs 4 --ckpt --inj controlnet --gan_start 100000' 't_enc_raster 45 --bs 4 --ckpt --enc raster --gan_start 100000' > /root/queue.txt
rm -f /root/runs/t_hands_none/last.pt
pid=$(pgrep -f "train.py --out /root/runs/t_hands_none" | head -1); kill $pid
sleep 25; tail -3 /root/q.log; cat /root/queue.txt | cut -c1-30
