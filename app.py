from flask import Flask, jsonify,request
import requests
import docker
import subprocess
import os
import time 

client = docker.from_env()

app = Flask(__name__)

@app.route('/docker/api/status')
def status(container_id=""):
    container_id = request.args.get('name')
    result =subprocess.run(['docker','ps', '--filter', 'status=running'], stdout=subprocess.PIPE)
    if str(result.stdout).find(container_id) >= 0 : 
        requests.post('http://localhost:3000/gpu', json={"status" :"Running", "time" :time.time()})
        return jsonify(status = "Running", time = time.time())
    else:
        requests.post('http://localhost:3000/gpu', json={"status" :"Stopped", "time" :time.time()})
        return jsonify(status = "Stopped", time = time.time())

@app.route('/docker/api/start')
def docker_run():
    client = docker.from_env()
    container = client.containers.run("neoanarika/mnist:latest",detach = True)
    return container.short_id

@app.route('/docker/api/stop')
def docker_stop():
    container = request.args.get('name')
    os.system("docker stop {}".format(container))
    return status(container)

if __name__ == '__main__':
    os.system("""figlet -f starwars "Valhalla" """)
    app.run(debug=True)