import os
import subprocess

def auto_serve(model, wait=True):
    serve_script = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm070.py"))
    cmd = f"python {serve_script} --model {model} --tp 1"
    print("serving command:")
    print(cmd)
    if wait:
        os.system(f"yes | serve shutdown -a http://localhost:{DASH_PORT}")
        # setsid so that it can be interrupted
        serve_p = subprocess.Popen(cmd.split(), preexec_fn=os.setsid)
        while True:
            print("try to conntect...")
            p = subprocess.run(["curl", "-m", "100000000", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
            if p.returncode != 0:
                print("waiting...")
                time.sleep(5)
            else:
                print("connected")
                break
    else:
        p = subprocess.run(["curl", "-m", "10", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
        if p.returncode != 0:
            print("server not started")
            exit(1)
    print(p.stdout)