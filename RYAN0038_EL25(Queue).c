#include <stdio.h>


void queue_init(void);
int r, f, val,i;
int SIZE=15;
int queue[15];

void main()
{
    queue_init();
    for(i=1;i<=15;i++)
    {
        printf("Input: ",i);
        scanf("%d",&val);

        if(val>0)
            enqueue(val);

        else if(val<0)
        {
          front(val);
           if (front(val)>0)
                printf("Output is: %d \n",front(val));
            else if (front(val)==0)
                printf("Error - Queue is empty.\n");
        }


        else if(val==0 && empty())
        {
            dequeue(val);
            printf("Error - Queue is empty.\n");
        }

        else if(val==0 && !empty())
        {
            dequeue(val);

        }
    }
}

void queue_init()
{
    r=f=-1;
}

int empty()
{
    return r == -1;

}

int enqueue(val)
{
    if (r == SIZE)
        return 0;
    else
    {


    if (empty())

    {
        r=f=0;
        queue[f]=val;

    }

    else
    {

        r=r+1;
        if(r==SIZE)
            r=0;
    }
    queue[r]= val;
    }return 1;

}

int dequeue()
{
    if (r==f)
    {
        r=f=-1;

    }

    else
    {
        f=f+1;
        if(f==SIZE)
            f=0;
    }
}

int front()
{
    return queue[f];
}

int rear()
{
    if(val == 0)
    {
        enqueue();
    }
    return queue[r];

}

