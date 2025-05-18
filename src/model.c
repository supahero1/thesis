/*
 *   Copyright 2025 Franciszek Balcerak
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thesis/debug.h>
#include <thesis/model.h>
#include <thesis/alloc_ext.h>

#include <assimp/scene.h>
#include <assimp/cimport.h>
#include <assimp/postprocess.h>


static const char*
const_basename(
	const char *path
	)
{
	const char* slash = strrchr(path, '/');
	return slash ? slash + 1 : path;
}


model_t*
model_init(
	const char* path
	)
{
	model_t* model = alloc_malloc(sizeof(*model));
	assert_not_null(model);

	const struct aiScene* scene = aiImportFile(
		path,
		aiProcess_GenNormals |
		aiProcess_SortByPType |
		aiProcess_GenUVCoords |
		aiProcess_Triangulate |
		aiProcess_OptimizeGraph |
		aiProcess_OptimizeMeshes |
		aiProcess_FindDegenerates |
		aiProcess_FindInvalidData |
		aiProcess_TransformUVCoords |
		aiProcess_FixInfacingNormals |
		aiProcess_PreTransformVertices |
		aiProcess_ImproveCacheLocality |
		aiProcess_JoinIdenticalVertices |
		aiProcess_ValidateDataStructure |
		aiProcess_RemoveRedundantMaterials
	);
	hard_assert_not_null(scene);

	assert_not_null(scene->mRootNode);
	assert_eq(scene->mNumCameras, 0);
	assert_eq(scene->mNumLights, 0);
	assert_eq(scene->mNumAnimations, 0);
	assert_gt(scene->mNumMaterials, 0);
	assert_gt(scene->mNumMeshes, 0);

	model->material_count = scene->mNumMaterials;
	model->materials = alloc_malloc(sizeof(*model->materials) * model->material_count);
	assert_not_null(model->materials);

	for(uint32_t i = 0; i < model->material_count; i++)
	{
		material_t* material = &model->materials[i];

		const struct aiMaterial* sceneMaterial = scene->mMaterials[i];
		assert_not_null(sceneMaterial);

		struct aiColor4D color;

		enum aiReturn status = aiGetMaterialColor(sceneMaterial, AI_MATKEY_COLOR_DIFFUSE, &color);
		assert_eq(status, AI_SUCCESS);
		glm_vec4_copy((void*) &color, material->diffuse);

		status = aiGetMaterialColor(sceneMaterial, AI_MATKEY_COLOR_AMBIENT, &color);
		assert_eq(status, AI_SUCCESS);
		glm_vec4_copy((void*) &color, material->ambient);

		material->texture = str_init();

		struct aiString path;
		if(AI_SUCCESS == aiGetMaterialTexture(sceneMaterial,
			aiTextureType_DIFFUSE, 0, &path, NULL, NULL, NULL, NULL, NULL, NULL))
		{
			str_set_copy_cstr(material->texture, const_basename(path.data));
		}
	}

	model->mesh_count = scene->mNumMeshes;
	model->meshes = alloc_malloc(sizeof(*model->meshes) * model->mesh_count);
	assert_not_null(model->meshes);

	for(uint32_t i = 0; i < model->mesh_count; i++)
	{
		mesh_t* mesh = &model->meshes[i];

		const struct aiMesh* sceneMesh = scene->mMeshes[i];
		assert_not_null(sceneMesh);

		assert_eq(sceneMesh->mPrimitiveTypes, aiPrimitiveType_TRIANGLE);
		assert_not_null(sceneMesh->mTextureCoords[0]);

		mesh->material_idx = sceneMesh->mMaterialIndex;
		mesh->vertex_count = sceneMesh->mNumVertices;
		assert_gt(mesh->vertex_count, 0);

		mesh->vertices = alloc_malloc(sizeof(*mesh->vertices) * mesh->vertex_count);
		assert_not_null(mesh->vertices);

		mesh->normals = alloc_malloc(sizeof(*mesh->normals) * mesh->vertex_count);
		assert_not_null(mesh->normals);

		mesh->coords = alloc_malloc(sizeof(*mesh->coords) * mesh->vertex_count);
		assert_not_null(mesh->coords);

		for(uint32_t j = 0; j < mesh->vertex_count; j++)
		{
			glm_vec3_copy((void*) &sceneMesh->mVertices[j], mesh->vertices[j]);
			glm_vec3_copy((void*) &sceneMesh->mNormals[j], mesh->normals[j]);
			glm_vec2_copy((void*) &sceneMesh->mTextureCoords[0][j], mesh->coords[j]);
		}

		mesh->index_count = sceneMesh->mNumFaces * 3;
		mesh->indexes = alloc_malloc(sizeof(*mesh->indexes) * mesh->index_count);
		assert_not_null(mesh->indexes);

		for(uint32_t j = 0; j < sceneMesh->mNumFaces; j++)
		{
			const struct aiFace* face = &sceneMesh->mFaces[j];
			assert_eq(face->mNumIndices, 3);

			mesh->indexes[j * 3 + 0] = face->mIndices[0];
			mesh->indexes[j * 3 + 1] = face->mIndices[1];
			mesh->indexes[j * 3 + 2] = face->mIndices[2];
		}
	}

	aiReleaseImport(scene);

	return model;
}


void
model_free(
	model_t* model
	)
{
	assert_not_null(model);

	for(uint32_t i = 0; i < model->mesh_count; i++)
	{
		mesh_t* mesh = &model->meshes[i];
		assert_not_null(mesh);

		alloc_free(mesh->indexes, sizeof(*mesh->indexes) * mesh->index_count);

		alloc_free(mesh->coords, sizeof(*mesh->coords) * mesh->vertex_count);
		alloc_free(mesh->normals, sizeof(*mesh->normals) * mesh->vertex_count);
		alloc_free(mesh->vertices, sizeof(*mesh->vertices) * mesh->vertex_count);
	}

	alloc_free(model->meshes, sizeof(*model->meshes) * model->mesh_count);

	for(uint32_t i = 0; i < model->material_count; i++)
	{
		material_t* material = &model->materials[i];
		assert_not_null(material);

		str_free(material->texture);
	}

	alloc_free(model->materials, sizeof(*model->materials) * model->material_count);

	alloc_free(model, sizeof(*model));
}
