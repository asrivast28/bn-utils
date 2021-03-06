/**
 * @file SetUtils.hpp
 * @brief Declaration of functions for common set operations.
 * @author Ankit Srivastava <asrivast@gatech.edu>
 *
 * Copyright 2020 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SETUTILS_HPP_
#define SETUTILS_HPP_

#include "mxx/comm.hpp"

#include <algorithm>
#include <iterator>
#include <vector>
#include <ostream>

/**
 * @brief Class that provides a lightweight subset wrapper over an STL container.
 *
 * @tparam Iterator The type of the iterator.
 *
 * This class allows iterating over a contiguous subset of a container as defined
 * by the first iterator, the last iterator, and the size of the slice.
 */
template <typename Iterator>
class SubsetWrapper {
public:
  SubsetWrapper(const Iterator first, const Iterator last, const uint32_t size)
    : m_begin(first),
      m_end(last),
      m_size(size)
  {
  }

  const Iterator&
  begin() const
  {
    return m_begin;
  }

  const Iterator&
  end() const
  {
    return m_end;
  }

  uint32_t
  size() const
  {
    return m_size;
  }

private:
  const Iterator m_begin;
  const Iterator m_end;
  const uint32_t m_size;
}; // class SubsetWrapper

/**
 * @brief Class that iterates over all the subsets of the given size of a given set.
 *
 * @tparam Set Type of the set container.
 * @tparam Element Type of the variable (expected to be an integral type).
 */
template <template <typename...> class SetType, typename Element, typename... Args>
class Subsets;

/**
 * @brief Function for initializing a given set.
 */
template <typename Set, typename Element>
Set
set_init(Set&&, const Element);

/**
 * @brief Function for checking if a given set contains a value.
 */
template <typename Set, typename Element>
bool
set_contains(const Set&, const Element);

/**
 * @brief Function for getting the union of two given sets.
 */
template <typename Set>
Set
set_union(const Set&, const Set&);

/**
 * @brief Function for getting the intersection of two given sets.
 */
template <typename Set>
Set
set_intersection(const Set&, const Set&);

/**
 * @brief Function for getting the difference of the second set from the first.
 */
template <typename Set>
Set
set_difference(const Set&, const Set&);

/**
 * @brief Function for broadcasting a set.
 */
template <typename Set>
void
set_bcast(Set&, const int, const mxx::comm&);

/**
 * @brief Function for broadcasting a vector of sets.
 */
template <typename Set, typename Var>
void
set_bcast(std::vector<std::reference_wrapper<Set>>&, const Var, const int, const mxx::comm&);

/**
 * @brief Function for set union across processes.
 */
template <typename Set>
void
set_allunion(Set&, const mxx::comm&);

/**
 * @brief Function for efficient union of multiple sets, done in accordance
 *        with their indices, across processes.
 */
template <typename Set, typename Var>
void
set_allunion_indexed(std::unordered_map<Var, Set>&, const Set&, const Var, const mxx::comm&);

/**
 * @brief Function for set intersection across processes.
 */
template <typename Set>
void
set_allintersect(Set&, const mxx::comm&);

/**
 * @brief Function for efficient intersection of multiple sets, done in accordance
 *        with their indices, across processes.
 */
template <typename Set, typename Var>
void
set_allintersect_indexed(std::unordered_map<Var, Set>&, const Set&, const Var, const mxx::comm&);

/**
 * @brief Function for gathering sets spread across processes.
 */
template <typename Set, typename Var>
std::vector<Set>
set_allgatherv(const std::vector<Set>&, const std::vector<size_t>&, const Var, const mxx::comm&);

#include "detail/StdSetUtils.hpp"
#include "detail/UintSetUtils.hpp"

#endif // SETUTILS_HPP_
