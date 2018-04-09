#include <vector>

/// <summary>
/// This class implements a <code>PriorityQueue</code>. This class
/// is implemented in such a way that objects are added using an
/// <code>add</code> function. The <code>add</code> function takes
/// two parameters an object and a long.
/// <p>
/// The object represents an item in the queue, the long indicates
/// its priority in the queue. The remove function in this class
/// returns the object first in the queue and that object is removed
/// from the queue permanently.
/// 
/// @author Simon Levy
/// Translation by Marco A. Alvarez
/// </summary>
template <typename T>
class PriorityQueue
{
private:
	/**
	* The maximum priority possible in this priority queue.
	*/
	double maxPriority;

	/**
	* This contains the list of objects in the queue.
	*/
	std::vector<T*> data;

	/**
	* This contains the list of prioritys in the queue.
	*/
	std::vector<double> value;

	/**
	* Holds the number of elements currently in the queue.
	*/
	int count;

	/**
	* This holds the number elements this queue can have.
	*/
	int capacity;

	/**
	* This is an initializer for the object. It basically initializes
	* an array of long called value to represent the prioritys of
	* the objects, it also creates an array of objects to be used
	* in parallel with the array of longs, to represent the objects
	* entered, these can be used to sequence the data.
	*
	* @param size the initial capacity of the queue, it can be
	* resized
	*/
	void init(int size)
	{
		maxPriority = DBL_MAX;
		count = 0;

		capacity = size;
		data = std::vector<T*>(capacity + 1);
		value = std::vector<double>(capacity + 1);
		value[0] = maxPriority;
		data[0] = nullptr;
	}

	/**
	* Bubble down is used to put the element at subscript 'pos' into
	* it's rightful place in the heap (i.e heap is another name
	* for <code>PriorityQueue</code>). If the priority of an element
	* at subscript 'pos' is less than it's children then it must
	* be put under one of these children, i.e the ones with the
	* maximum priority must come first.
	*
	* @param pos is the position within the arrays of the element
	* and priority
	*/
	void bubbleDown(int pos)
	{
		T* element = data[pos];
		double priority = value[pos];
		int child;
		/* hole is position '1' */
		for (; pos * 2 <= count; pos = child)
		{
			child = pos * 2;
			/* if 'child' equals 'count' then there
			is only one leaf for this parent */
			if (child != count)

				/* left_child > right_child */
			if (value[child] < value[child + 1])
				child++; /* choose the biggest child */
			/* percolate down the data at 'pos', one level
			i.e biggest child becomes the parent */
			if (priority < value[child])
			{
				value[pos] = value[child];
				data[pos] = data[child];
			}
			else
			{
				break;
			}
		}
		value[pos] = priority;
		data[pos] = element;
	}

	/**
	* Bubble up is used to place an element relatively low in the
	* queue to it's rightful place higher in the queue, but only
	* if it's priority allows it to do so, similar to bubbleDown
	* only in the other direction this swaps out its parents.
	*
	* @param pos the position in the arrays of the object
	* to be bubbled up
	*/
	void bubbleUp(int pos)
	{
		T* element = data[pos];
		double priority = value[pos];
		/* when the parent is not less than the child, end*/
		while (value[pos / 2] < priority)
		{
			/* overwrite the child with the parent */
			value[pos] = value[pos / 2];
			data[pos] = data[pos / 2];
			pos /= 2;
		}
		value[pos] = priority;
		data[pos] = element;
	}

	/**
	* This ensures that there is enough space to keep adding elements
	* to the priority queue. It is however advised to make the capacity
	* of the queue large enough so that this will not be used as it is
	* an expensive method. This will copy across from 0 as 'off' equals
	* 0 is contains some important data.
	*/
	void expandCapacity()
	{
		capacity = count * 2;
		std::vector<T*> elements = std::vector<T*>(count + 1);
		std::vector<double> prioritys = std::vector<double>(count + 1);

		data.insert(data.end(), elements.begin(), elements.end());
		value.insert(value.end(), prioritys.begin(), prioritys.end());
	}
public:
	/**
	* Creates a new <code>PriorityQueue</code> object. The
	* <code>PriorityQueue</code> object allows objects to be
	* entered into the queue and to leave in the order of
	* priority i.e the highest priority get's to leave first.
	*/
	PriorityQueue()
	{
		init(20);
	}

	/**
	* Creates a new <code>PriorityQueue</code> object. The
	* <code>PriorityQueue</code> object allows objects to
	* be entered into the queue an to leave in the order of
	* priority i.e the highest priority get's to leave first.
	*
	* @param capacity the initial capacity of the queue before
	* a resize
	*/
	PriorityQueue(int capacity)
	{
		init(capacity);
	}

	/**
	* Creates a new <code>PriorityQueue</code> object. The
	* <code>PriorityQueue</code> object allows objects to
	* be entered into the queue an to leave in the order of
	* priority i.e the highest priority get's to leave first.
	*
	* @param capacity the initial capacity of the queue before
	* a resize
	* @param maxPriority is the maximum possible priority for
	* an object
	*/
	PriorityQueue(int capacity, double maxPriority)
	{
		init(capacity);
		this->maxPriority = maxPriority;
	}

	/**
	* This function adds the given object into the <code>PriorityQueue</code>,
	* its priority is the long priority. The way in which priority can be
	* associated with the elements of the queue is by keeping the priority
	* and the elements array entrys parallel.
	*
	* @param element is the object that is to be entered into this
	* <code>PriorityQueue</code>
	* @param priority this is the priority that the object holds in the
	* <code>PriorityQueue</code>
	*/
	void add(T* element, double priority)
	{
		if (count++ >= capacity)
		{
			expandCapacity();
		}
		/* put this as the last element */
		value[count] = priority;
		data[count] = element;
		bubbleUp(count);
	}

	/**
	* Remove is a function to remove the element in the queue with the
	* maximum priority. Once the element is removed then it can never be
	* recovered from the queue with further calls. The lowest priority
	* object will leave last.
	*
	* @return the object with the highest priority or if it's empty
	* null
	*/
	T* remove()
	{
		if (count == 0)
			return nullptr;
		T* element = data[1];
		/* swap the last element into the first */
		data[1] = data[count];
		value[1] = value[count];
		/* let the GC clean up */
		data[count] = nullptr;
		value[count] = 0L;
		count--;
		bubbleDown(1);
		return element;
	}

	T* front()
	{
		return data[1];
	}

	double getMaxPriority()
	{
		return value[1];
	}

	/**
	* This method will empty the queue. This also helps garbage
	* collection by releasing any reference it has to the elements
	* in the queue. This starts from offset 1 as off equals 0
	* for the elements array.
	*/
	void clear()
	{
		for (int i = 1; i < count; i++)
		{
			data[i] = nullptr; /* help gc */
		}
		count = 0;
	}

	/**
	* The number of elements in the queue. The length
	* indicates the number of elements that are currently
	* in the queue.
	*
	* @return the number of elements in the queue
	*/
	int length()
	{
		return count;
	}
};

template <typename T>
class NearestNeighborList
{
public:
	int REMOVE_HIGHEST;
	int REMOVE_LOWEST;

	PriorityQueue<T> m_Queue;
	int m_Capacity;

	// constructor
	NearestNeighborList(int capacity)
	{
		REMOVE_HIGHEST = 1;
		REMOVE_LOWEST = 2;

		m_Capacity = capacity;
		m_Queue = PriorityQueue<T>(m_Capacity, DBL_MAX);
	}

	double getMaxPriority()
	{
		if (m_Queue.length() == 0)
		{
			return DBL_MAX;
		}
		return m_Queue.getMaxPriority();
	}

	bool insert(T* _object, double priority)
	{
		if (m_Queue.length() < m_Capacity)
		{
			// capacity not reached
			m_Queue.add(_object, priority);
			return true;
		}
		if (priority > m_Queue.getMaxPriority())
		{
			// do not insert - all elements in queue have lower priority
			return false;
		}
		// remove object with highest priority
		m_Queue.remove();
		// add new object
		m_Queue.add(_object, priority);
		return true;
	}

	bool isCapacityReached()
	{
		return m_Queue.length() >= m_Capacity;
	}

	T* getHighest()
	{
		return m_Queue.front();
	}

	bool isEmpty()
	{
		return m_Queue.length() == 0;
	}

	int getSize()
	{
		return m_Queue.length();
	}

	T* removeHighest()
	{
		// remove object with highest priority
		return m_Queue.remove();
	}
};

class HRect
{
public:
	std::vector<double> min;
	std::vector<double> max;

	HRect()
	{
	}

	HRect(int ndims)
	{
		min = std::vector<double>(ndims);
		max = std::vector<double>(ndims);
	}

	HRect(std::vector<double> vmin, std::vector<double> vmax)
	{

		min = vmin;
		max = vmax;
	}

	HRect clone()
	{

		return HRect(min, max);
	}

	std::vector<double> closest(std::vector<double> t)
	{

		std::vector<double> p = std::vector<double>(t.size());

		for (unsigned int i = 0; i < t.size(); ++i)
		{
			if (t[i] <= min[i])
			{
				p[i] = min[i];
			}
			else if (t[i] >= max[i])
			{
				p[i] = max[i];
			}
			else
			{
				p[i] = t[i];
			}
		}

		return p;
	}

	static HRect infiniteHRect(int d)
	{

		std::vector<double> vmin = std::vector<double>(d);
		std::vector<double> vmax = std::vector<double>(d);

		for (int i = 0; i < d; ++i)
		{
			vmin[i] = -DBL_MAX;
			vmax[i] = DBL_MAX;
		}

		return HRect(vmin, vmax);
	}
};

template <typename L>
class KDNode
{
public:
	std::vector<double> _k;
	L _v;
	KDNode* _left, *_right;
	bool _deleted;

	// constructor is used only by class; other methods are static
	KDNode(std::vector<double> key, L val)
	{
		_k = key;
		_v = val;
		_left = nullptr;
		_right = nullptr;
		_deleted = false;
	}

	static bool equals(std::vector<double> k1, std::vector<double> k2)
	{
		for (unsigned int i = 0; i < k1.size(); ++i)
		if (k1[i] != k2[i])
			return false;

		return true;
	}

	static double sqrdist(std::vector<double> x, std::vector<double> y)
	{
		double dist = 0;

		for (unsigned int i = 0; i < x.size(); ++i)
		{
			double diff = (x[i] - y[i]);
			dist += (diff * diff);
		}

		return dist;

	}

	static double eucdist(std::vector<double> x, std::vector<double> y)
	{
		return std::sqrt(sqrdist(x, y));
	}

	// Method insert 
	static KDNode* ins(std::vector<double> key, L val, KDNode *t, int lev, int K)
	{
		if (t == nullptr)
		{
			t = new KDNode(key, val);
		}
		else if (equals(key, t->_k))
		{
			// "re-insert"
			if (t->_deleted)
			{
				t->_deleted = false;
				t->_v = val;
			}
			//else
			//{
			//	//return nullptr;//throw (new KeyDuplicateException());
			//}
		}

		else if (key[lev] > t->_k[lev])
		{
			t->_right = ins(key, val, t->_right, (lev + 1) % K, K);
		}
		else
		{
			t->_left = ins(key, val, t->_left, (lev + 1) % K, K);
		}

		return t;
	}

	static KDNode* NodeDelete(std::vector<double> key, KDNode* t, int lev, int K, bool& deleted) {
		if (t == nullptr)
			return nullptr;
		if (!t->_deleted && equals(key, t->_k))
		{
			t->_deleted = true;
			deleted = true;

			delete t->_v;
			t->_v = nullptr;
		}
		else if (key[lev] > t->_k[lev])
		{
			t->_right = NodeDelete(key, t->_right, (lev + 1) % K, K, deleted);
		}
		else
		{
			t->_left = NodeDelete(key, t->_left, (lev + 1) % K, K, deleted);
		}

		if (!t->_deleted || t->_left != nullptr || t->_right != nullptr)
		{
			return t;
		}
		else
		{
			return nullptr;
		}
	}

	static void deleteNodes(KDNode *t)
	{
		if (t == nullptr)
		{
			return;
		}
		else
		{
			//delete t->_v;
			deleteNodes(t->_right);
			deleteNodes(t->_left);
		}

		return;
	}

	// Method Nearest Neighbor from Andrew Moore's thesis. Numbered
	// comments are direct quotes from there. Step "SDL" is added to
	// make the algorithm work correctly.  NearestNeighborList solution
	// courtesy of Bjoern Heckel.
	static void nnbr(KDNode* kd, std::vector<double>* target, HRect* hr,
		double max_dist_sqd, int lev, int K,
		NearestNeighborList<KDNode>* nnl)
	{

		// 1. if kd is empty then set dist-sqd to infinity and exit.
		if (kd == nullptr)
		{
			return;
		}

		// 2. s := split field of kd
		int s = lev % K;

		// 3. pivot := dom-elt field of kd
		std::vector<double> pivot = kd->_k;
		double pivot_to_target = sqrdist(pivot, *target);

		// 4. Cut hr into to sub-hyperrectangles left-hr and right-hr.
		//    The cut plane is through pivot and perpendicular to the s
		//    dimension.
		HRect left_hr = *hr; // optimize by not cloning
		HRect right_hr = hr->clone();
		left_hr.max[s] = pivot[s];
		right_hr.min[s] = pivot[s];

		// 5. target-in-left := target_s <= pivot_s
		bool target_in_left = (*target)[s] < pivot[s];

		KDNode* nearer_kd;
		HRect nearer_hr;
		KDNode* further_kd;
		HRect further_hr;

		// 6. if target-in-left then
		//    6.1. nearer-kd := left field of kd and nearer-hr := left-hr
		//    6.2. further-kd := right field of kd and further-hr := right-hr
		if (target_in_left)
		{
			nearer_kd = kd->_left;
			nearer_hr = left_hr;
			further_kd = kd->_right;
			further_hr = right_hr;
		}
		//
		// 7. if not target-in-left then
		//    7.1. nearer-kd := right field of kd and nearer-hr := right-hr
		//    7.2. further-kd := left field of kd and further-hr := left-hr
		else
		{
			nearer_kd = kd->_right;
			nearer_hr = right_hr;
			further_kd = kd->_left;
			further_hr = left_hr;
		}

		// 8. Recursively call Nearest Neighbor with paramters
		//    (nearer-kd, target, nearer-hr, max-dist-sqd), storing the
		//    results in nearest and dist-sqd
		nnbr(nearer_kd, target, &nearer_hr, max_dist_sqd, lev + 1, K, nnl);

		KDNode* nearest = (KDNode*)nnl->getHighest();
		double dist_sqd;

		if (!nnl->isCapacityReached())
		{
			dist_sqd = DBL_MAX;
		}
		else
		{
			dist_sqd = nnl->getMaxPriority();
		}

		// 9. max-dist-sqd := minimum of max-dist-sqd and dist-sqd
		max_dist_sqd = std::min(max_dist_sqd, dist_sqd);

		// 10. A nearer point could only lie in further-kd if there were some
		//     part of further-hr within distance sqrt(max-dist-sqd) of
		//     target.  If this is the case then
		std::vector<double> closest = further_hr.closest(*target);
		if (eucdist(closest, *target) < std::sqrt(max_dist_sqd))
		{

			// 10.1 if (pivot-target)^2 < dist-sqd then
			if (pivot_to_target < dist_sqd)
			{

				// 10.1.1 nearest := (pivot, range-elt field of kd)
				nearest = kd;

				// 10.1.2 dist-sqd = (pivot-target)^2
				dist_sqd = pivot_to_target;

				// add to nnl
				if (!kd->_deleted)
				{
					nnl->insert(kd, dist_sqd);
				}

				// 10.1.3 max-dist-sqd = dist-sqd
				// max_dist_sqd = dist_sqd;
				if (nnl->isCapacityReached())
				{
					max_dist_sqd = nnl->getMaxPriority();
				}
				else
				{
					max_dist_sqd = DBL_MAX;
				}
			}

			// 10.2 Recursively call Nearest Neighbor with parameters
			//      (further-kd, target, further-hr, max-dist_sqd),
			//      storing results in temp-nearest and temp-dist-sqd
			nnbr(further_kd, target, &further_hr, max_dist_sqd, lev + 1, K, nnl);
			KDNode* temp_nearest = (KDNode*)nnl->getHighest();
			double temp_dist_sqd = nnl->getMaxPriority();

			// 10.3 If tmp-dist-sqd < dist-sqd then
			if (temp_dist_sqd < dist_sqd)
			{

				// 10.3.1 nearest := temp_nearest and dist_sqd := temp_dist_sqd
				nearest = temp_nearest;
				dist_sqd = temp_dist_sqd;
			}
		}

		// SDL: otherwise, current point is nearest
		else if (pivot_to_target < max_dist_sqd)
		{
			nearest = kd;
			dist_sqd = pivot_to_target;
		}
	}
};

template <typename L>
class KDTree
{
public:
	KDTree(int k)
	{
		m_K = k;
		m_root = nullptr;
		m_count = 0;
	}

	/**
	* Find KD-tree node whose key is nearest neighbor to
	* key. Implements the Nearest Neighbor algorithm (Table 6.4) of
	*
	* @param key key for KD-tree node
	*
	* @return object at node nearest to key, or null on failure
	*
	* @throws KeySizeException if key.length mismatches K

	*/
	L nearest(std::vector<double> key)//, std::vector<double>& ks)
	{
		//std::vector<std::vector<double>> Ks;
		std::vector<L> nbrs = nearest(key, 1);//, Ks);
		if (nbrs.size() == 0)
		{
			//		ks = std::vector<double>();
			return -1;
		}
		//			ks = Ks[0];
		return nbrs[0];
	}

	/**
	* Find KD-tree nodes whose keys are <I>n</I> nearest neighbors to
	* key. Uses algorithm above.  Neighbors are returned in ascending
	* order of distance to key.
	*
	* @param key key for KD-tree node
	* @param n how many neighbors to find
	*
	* @return objects at node nearest to key, or null on failure
	*
	* @throws KeySizeException if key.length mismatches K
	* @throws IllegalArgumentException if <I>n</I> is negative or
	* exceeds tree size
	*/
	std::vector<L> nearest(std::vector<double> key, int n)//, std::vector<std::vector<double>>& Ks)
	{

		if (n < 0)
		{
			return std::vector<L>();// throw new ArgumentException("Number of neighbors cannot be negative or greater than number of nodes");
		}
		if (n > m_count)
		{
			n = m_count;
		}

		if (n == 0)
		{
			return std::vector<L>();
		}

		if (key.size() != m_K)
		{
			return std::vector<L>(); //throw new KeySizeException();
		}

		std::vector<L> nbrs = std::vector<L>();
		NearestNeighborList<KDNode<L>> nnl(n);

		// initial call is with infinite hyper-rectangle and max distance
		HRect hr = HRect::infiniteHRect(key.size());
		double max_dist_sqd = DBL_MAX;

		KDNode<L>::nnbr(m_root, &key, &hr, max_dist_sqd, 0, m_K, &nnl);

		for (int i = 0; i < n; ++i)
		{
			KDNode<L>* kd = (KDNode<L>*)nnl.removeHighest();
			nbrs.push_back(kd->_v);
			//			Ks.push_back(kd->_k);
		}

		return nbrs;
	}

	/**
	* Insert a node in a KD-tree.
	*
	* @param key key for KD-tree node
	* @param value value at that key
	*
	* @throws KeySizeException if key.length mismatches K
	* @throws KeyDuplicateException if key already in tree
	*/
	void insert(std::vector<double> key, L value)
	{

		if (key.size() != m_K)
		{
			return;
		}
		else
		{
			m_root = KDNode<L>::ins(key, value, m_root, 0, m_K);
		}

		m_count++;
	}

	bool deleteNode(std::vector<double> key)
	{

		if (key.size() != m_K)
		{
			return false;
		}

		else
		{
			bool deleted = false;
			m_root = KDNode<L>::NodeDelete(key, m_root, 0, m_K, deleted);
			if (deleted == false)
			{
				return false; // throw new KeyMissingException();
			}
			m_count--;
		}
		return true;
	}

	void clear()
	{
		KDNode<L>::deleteNodes(m_root);
	}

private:
	// K = number of dimensions
	int m_K;

	// root of KD-tree
	KDNode<L>* m_root;

	// count of nodes
	int m_count;
};