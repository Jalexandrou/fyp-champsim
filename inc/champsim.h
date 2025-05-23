/*
 *    Copyright 2023 The ChampSim Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CHAMPSIM_H
#define CHAMPSIM_H

#include <cstdint>
#include <exception>

namespace champsim
{
struct deadlock : public std::exception {
  const uint32_t which;
  explicit deadlock(uint32_t cpu) : which(cpu) {}
};

#ifdef DEBUG_PRINT
constexpr bool debug_print = true;
#else
constexpr bool debug_print = false;
#endif

#ifdef BOP_DBUG
constexpr bool bop_debug = true;
#else
constexpr bool bop_debug = false;
#endif

#ifdef MULTI_BOP_DBUG
constexpr bool multi_bop_dbug = true;
#else
constexpr bool multi_bop_dbug = false;
#endif

#ifdef CAERUS_DBUG
constexpr bool caerus_dbug = true;
#else
constexpr bool caerus_dbug = false;
#endif

#ifdef TEST_DBUG
constexpr bool test_dbug = true;
#else
constexpr bool test_dbug = false;
#endif
} // namespace champsim

#endif
