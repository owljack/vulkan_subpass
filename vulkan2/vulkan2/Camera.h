#pragma once
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include <glm/common.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <stdexcept>

//Holds information about Camera to view the scene through. Should probably only be 1 of these(for the moment).
class Camera
{
public:
	Camera(glm::vec3 pos, glm::vec3 lookAt, glm::vec3 up, float width, float height) :
		cameraPos(pos), cameraUp(up), cameraLookingAt(lookAt), viewMatrix(glm::mat4(1.0f)), perspectiveMatrix(glm::mat4(1.0f)), windowWidth(width), windowHeight(height)
	{
		viewMatrix = glm::lookAt(cameraPos, cameraLookingAt, cameraUp);
		perspectiveMatrix = glm::perspective(glm::radians(45.0f), width/height, 1.0f, 20.0f);
	}

	glm::mat4 GetViewMatrix() const
	{
		return viewMatrix;
	}

	glm::mat4 GetPerspectiveMatrix() const
	{
		return perspectiveMatrix;
	}

	glm::vec3 GetCurrentPosition() const
	{
		return cameraPos;
	}

	void SetCurrentPosition(const glm::vec3& pos)
	{
		cameraPos = pos;
		viewMatrix = glm::lookAt(cameraPos, cameraLookingAt, cameraUp);
	}

	void MoveCamera(glm::vec3 movePos)
	{
		cameraPos += movePos;
		viewMatrix = glm::lookAt(cameraPos, cameraLookingAt, cameraUp);
	}

	void RotateCamera()
	{
		throw std::runtime_error("Function Not Implemented!");
	}

	void ResetCameraToPosition(glm::vec3 pos)
	{
		cameraPos = pos;
		viewMatrix = glm::lookAt(cameraPos, cameraLookingAt, cameraUp);
	}

private:
	glm::vec3 cameraPos;
	glm::vec3 cameraLookingAt;
	glm::vec3 cameraUp;
	glm::mat4 viewMatrix;
	glm::mat4 perspectiveMatrix;
	float windowWidth;
	float windowHeight;
};
