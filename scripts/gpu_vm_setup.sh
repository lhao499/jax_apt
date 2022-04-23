#! /bin/bash

sudo apt-get update && sudo apt-get install -y \
    build-essential \
    tmux \
    htop \
    git \
    nodejs \
    bmon \
    p7zip-full \
    nfs-common


cat > $HOME/gpu_requirements.txt <<- EndOfFile
-f https://storage.googleapis.com/jax-releases/jax_releases.html
jax[cuda]==0.3.4
flax==0.4.0
optax==0.1.1
distrax==0.1.1
brax==0.0.12
tqdm
cloudpickle==2.0.0
dill
ml_collections
wandb==0.12.7
scikit-image==0.19.2
gcsfs==2022.02.0
-f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
torch==1.8.2+cpu
torchvision==0.9.2+cpu
EndOfFile

pip install --no-cache-dir -r $HOME/gpu_requirements.txt --user


# VIM configurations
cat > $HOME/.vimrc <<- EndOfFile
set tabstop=4
set shiftwidth=4
set softtabstop=4
set expandtab
set backspace=indent,eol,start
syntax on
EndOfFile

# Tmux configurations
cat > $HOME/.tmux.conf <<- EndOfFile
bind r source-file ~/.tmux.conf \; display-message "█▓░ ~/.tmux.conf reloaded."

# start with window 1 (instead of 0)
set -g base-index 1

set -g prefix C-a

set -g set-titles on
set -g set-titles-string '#(whoami)::#h::#(curl ipecho.net/plain;echo)'

# Status bar customization
#set -g status-utf8 on
set -g status-bg white
set -g status-fg black
set -g status-interval 5
set -g status-left-length 90
set -g status-right-length 60
set -g status-justify left

# send the prefix to client inside window (ala nested sessions)
bind-key a send-prefix

bind-key x kill-pane

# auto reorder
set-option -g renumber-windows on

# default statusbar colors
set-option -g status-style fg=yellow,bg=white #yellow and base2

# default window title colors
set-window-option -g window-status-style fg=brightyellow,bg=default #base0 and default

# active window title colors
set-window-option -g window-status-current-style fg=brightred,bg=default #orange and default

# pane border
set-option -g pane-border-style fg=white #base2
set-option -g pane-active-border-style fg=brightcyan #base1
EndOfFile


# HTop Configurations
mkdir -p $HOME/.config/htop
cat > $HOME/.config/htop/htoprc <<- EndOfFile
# Beware! This file is rewritten by htop when settings are changed in the interface.
# The parser is also very primitive, and not human-friendly.
fields=0 48 17 18 38 39 40 2 46 47 49 1
sort_key=46
sort_direction=1
hide_threads=0
hide_kernel_threads=1
hide_userland_threads=1
shadow_other_users=0
show_thread_names=0
show_program_path=1
highlight_base_name=0
highlight_megabytes=1
highlight_threads=1
tree_view=0
header_margin=1
detailed_cpu_time=0
cpu_count_from_zero=0
update_process_names=0
account_guest_in_cpu_meter=0
color_scheme=0
delay=15
left_meters=CPU Memory Swap
left_meter_modes=1 1 1
right_meters=Tasks LoadAverage Uptime
right_meter_modes=2 2 2
EndOfFile