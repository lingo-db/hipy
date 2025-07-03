#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

namespace ds {
    template<class V>
    class ChunkedBuffer {
        size_t currCapacity;
        size_t totalLength;

        struct Chunk {
            V* data;
            size_t size;
        };

        std::vector<Chunk> chunks;

        void nextBuffer() {
            size_t newCapacity = currCapacity * 2;
            chunks.push_back({new V[newCapacity], 0});
            currCapacity = newCapacity;
        }

    public:
        ChunkedBuffer(size_t capacity = 1024):currCapacity(capacity), totalLength(0){
            chunks.push_back({new V[capacity], 0});
        }

        V* allocate() {
            if (chunks.back().size == currCapacity) {
                nextBuffer();
            }
            totalLength++;
            return chunks.back().data + chunks.back().size++;
        }

        template<class F>
        void iterate(const F& f) {
            for (auto& chunk: chunks) {
                for (size_t i = 0; i < chunk.size; i++) {
                    f(chunk.data[i]);
                }
            }
        }
        size_t size() {
            return totalLength;
        }
        ~ChunkedBuffer() {
            for (auto& chunk: chunks) {
                delete[] chunk.data;
            }
        }
    };

    template<class K, class V>
    class JoinHashTable {
        static uint64_t nextPow2(uint64_t v) {
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v |= v >> 32;
            v++;
            return v;
        }
        struct Entry {
            Entry* next;
            uint64_t hash;
            K key;
            V value;
        };
        Entry** table;
        ChunkedBuffer<Entry> buffer;
        size_t htMask;
    public:
        void insert(K k, V v) {
            std::hash<K> hash_fn;
            uint64_t hash = hash_fn(k);
            Entry* allocated= buffer.allocate();
            allocated->next=nullptr;
            allocated->hash=hash;
            allocated->key=k;
            allocated->value=v;
        }
        void build() {
            size_t htSize = nextPow2(buffer.size());
            htMask = htSize - 1;
            table = new Entry*[htSize];
            for (size_t i = 0; i < htSize; i++) {
                table[i] = nullptr;
            }
            buffer.iterate([this](Entry& e) {
                size_t idx = e.hash & htMask;
                e.next = table[idx];
                table[idx] = &e;
            });
        }
        struct Iterator {
            size_t hash;
            K key;
            Entry* curr;
            V& current() {
                return curr->value;
            }
            bool valid() {
                return curr != nullptr;
            }
            void next() {
                while(curr != nullptr) {
                    curr = curr->next;
                    if (curr&&curr->hash==hash && curr->key==key) {
                        break;
                    }
                }
            }

        };
        Iterator find(K k) {
            std::hash<K> hash_fn;
            uint64_t hash = hash_fn(k);
            size_t idx = hash & htMask;
            Entry* e = table[idx];
            while (e != nullptr) {
                if (e->hash == hash && e->key == k) {
                    return Iterator{.hash=hash, .key=k, .curr=e};
                }
                e = e->next;
            }
            return Iterator{.hash=hash, .key=k, .curr=nullptr};
        }
        ~JoinHashTable() {
            delete[] table;
        }


    };
}
#endif //DATASTRUCTURES_H
