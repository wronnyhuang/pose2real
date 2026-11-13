[ -f /root/FIX1 ] && { cat /root/queue.txt | cut -c1-40; exit 0; }
touch /root/FIX1
grep -q '^t_v3_seen' /root/queue.txt || { echo "v3 already started?"; }
sed -i '/^t_hand_gan/d' /root/queue.txt
echo 't_v4_long 90 --bs 4 --ckpt --audio win --au_la -1 --refbase --w_lips 3 --resume /root/runs/t_v3_seen/last.pt --splits train,heldout_ref --boost 8 --gan_start 100000' >> /root/queue.txt
cat /root/queue.txt | cut -c1-40
