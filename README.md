# llama-3.2-1b.vb
llama 3.2 1b fp16 cpu inference in one file of pure VB.NET

I used Gemini 2.5 Pro Preview to port this C++ repo: https://github.com/iangitonga/llama32.cpp/tree/3a4283044a652f53e71710780456d2d5c8a8bc9a to VB.NET. So all credits go to creator of original repo and to Gemini 2.5 Pro Preview.
You should manually download this file: https://huggingface.co/iangitonga/llama32/resolve/main/llama32-1B.fp16.bin and move it to models subfolder. Models subfolder should be created in the folder where you placed app's release files. You should also put tokenizer file https://github.com/iangitonga/llama32.cpp/blob/3a4283044a652f53e71710780456d2d5c8a8bc9a/tokenizer.bin to app's release folder.

You can build project by double clicking on build.bat
