#!/usr/bin/env python3

#   Copyright 2025 Franciszek Balcerak
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os


env = Environment(tools = ["mingw"] if os.name == "nt" else ["default"], ENV=os.environ)

flags = Split("-std=gnu23 -Wall -Iinclude/ -D_GNU_SOURCE")

release = int(ARGUMENTS["RELEASE"] if "RELEASE" in ARGUMENTS else os.environ.get("RELEASE", "0"))
if release <= 0:
	flags.extend(Split("-O0 -g3"))
	if os.name != "nt":
		env.Append(LINKFLAGS=Split("-rdynamic"))
else:
	if release >= 1:
		flags.extend(Split("-O3 -DNDEBUG -flto"))
	if release >= 2:
		flags.extend(Split("-march=native"))

env.Append(CPPFLAGS=flags)

libs = Split("m SDL3 assimp openxr_loader")
if os.name == "nt":
	libs.extend(Split("vulkan-1 ws2_32"))
else:
	libs.extend(Split("vulkan"))
env.Append(LIBS=libs)



def help(target, source, env):
	print("""
Specify one (or more) of the following:

app         generates the app

Specify RELEASE=1 for a production build.
Specify RELEASE=2 for a native build (faster than production but not portable).
	""")

env.AlwaysBuild(env.Alias("help", [], help))

Default("help")



deps = {}

def prop_headers(dest, headers):
	for header in headers:
		if header not in dest:
			dest.append(header)
			if header in deps:
				prop_headers(dest, deps[header])

def add_file(path):
	with open(path, "r") as f:
		if f.readline().startswith("/* skip */"):
			return False
		include_lines = [line for line in f if line.startswith("#include <thesis/")]
		headers = ["include/" + line[10:-2] for line in include_lines]
		deps[path] = headers
		for header in headers:
			add_file(header)
		for header in headers:
			if header in deps:
				prop_headers(headers, deps[header])
	return True

def add_files(path):
	output = []
	for root, _, files in os.walk(path):
		for file in files:
			if file.endswith(".c"):
				filepath = os.path.join(root, file)
				if add_file(filepath):
					output.append(filepath)
	return output

app_files = add_files("src")



objects = {}

def add_object(file):
	files = deps[file]
	obj = env.Object("bin/" + file[:-1] + "o", file)[0]
	objects[file] = obj
	for file in files:
		if file.endswith(".c"):
			if file not in objects:
				add_object(file)
			env.Depends(obj, objects[file])
		else:
			env.Depends(obj, file)
	return obj

def add_objects(files):
	return [add_object(file) for file in files]

app_objects = add_objects(app_files)

app = env.Program("bin/app", app_objects)

env.Alias("app", app)
