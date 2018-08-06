


#pragma once
#include <glm/glm.hpp>

class Octree
{
public:
	//Axis Aligned Bounding Box
	struct AABB
	{
		AABB();
		~AABB();

		glm::fmat2x3 m_box;
		glm::fvec3   m_center;		
	};

	enum NodeType{NT_LEAF, NT_INTERNEL, NT_EMPTY, NT_EMPTY}
	struct Node
	{
		Node();
		~Node();

		AABB  	m_box;
		float 	m_singed_dist;	//The distance to surface
		int   	m_node_type;
		Node* 	m_parent;
		Node* 	m_children[8];
	};

	Octree(float voxel_size = 1) : m_voxel_size(voxel_size),
	                               m_root(NULL),
	                               m_num_nodes(0),
	                               m_num_leafs(0),
	                               m_num_externals(0)
	{}
	~Octree();

	void initTree();

	Node* getRoot()
	{
		return m_root;
	}
	void setRoot(Node* root) : m_root(root)
	{}

	//TODO: 
	//1. add getters and setters

private:
	Node*     	m_root;
	glm::fvec3 	m_origin;		//The lower left corner
	float      	m_voxel_size;	//length of a side of a voxel, (default to 1)
	uint       	m_num_nodes;
	uint       	m_num_leafs;
	uint       	m_num_internals;
	uint       	m_num_externals;


};